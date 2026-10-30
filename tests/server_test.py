# /// script
# requires-python = ">=3.12"
# dependencies = ["natlas-core", "protobuf>=7.36.1", "starlette>=0.40", "httpx2>=2.0", "uvicorn>=0.30"]
# [tool.uv.sources]
# natlas-core = { path = "../core" }
# ///
"""Native core and real Connect transport tests without loading an ML model.

--serve starts the same deterministic fixture for browser verification.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import service  # noqa: E402
from natlas_codec import labels_json, pb  # noqa: E402
from natlas_core import Tree, encode  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


class Values:
    def __init__(self, values):
        self.values = values

    def sort(self, descending=True):
        order = sorted(
            range(len(self.values)), key=self.values.__getitem__, reverse=descending
        )
        return Values([self.values[index] for index in order]), Values(order)

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Measure:
    name = "fixture"

    def __init__(self, weights=(0.6, 0.25, 0.1, 0.049, 0.001)):
        self.weights = weights
        self.calls = []

    def log_probs(self, path):
        self.calls.append(tuple(path))
        return Values([math.log(weight) for weight in self.weights])


def fixture():
    measures = [Measure(), Measure((0.25, 0.6, 0.1, 0.049, 0.001))]
    model = service.Model((99,), lambda path: len(path) >= 5, 5, measures)
    manifest = [
        {
            "name": "fixture",
            "title": "Integer sequence fixture",
            "labels": "labels.json",
            "measures": [
                {"name": f"measure {index}", "url": f"{index}/"} for index in range(2)
            ],
        }
    ]
    labels = [b"a", b"b", b"c", b"r", b"\n"]

    def search(text):
        return [b"abcr\n".index(character) for character in text.encode()]

    return model, service.app(
        model,
        manifest,
        {"labels.json": (labels_json(labels).encode(), "application/json")},
        search,
    )


def messages(body):
    result = []
    while body:
        flags, length = struct.unpack_from(">BI", body)
        payload, body = body[5 : 5 + length], body[5 + length :]
        result.append((flags, payload))
    return result


class CoreTests(unittest.TestCase):
    def test_codec_matches_browser_fixture(self):
        self.assertEqual(
            encode(
                [7, 3, 9, 1, 4, 2],
                [math.log(w) for w in [0.5, 0.3, 0.3, 0.15, 0.001, 1e-6]],
                12,
            ),
            [(7, 0), (3, 104683117), (9, 2147483647), (1, 0), (4, -1291920529)],
        )
        self.assertEqual(encode([0], [0.0], 0), [])

    def test_tree_paths_cache_and_worker_threads(self):
        tree = Tree(2)
        node = tree.node([0xFFFFFFFF, 7])
        self.assertEqual(tree.node([0xFFFFFFFF, 7]), node)
        tree.set_distribution(node, 1, [3, 8], [math.log(0.75), math.log(0.25)])
        with ThreadPoolExecutor() as pool:
            edges = pool.submit(tree.expand, node, 1, 16).result()
        self.assertEqual([label for label, _ in edges], [3, 8])
        self.assertEqual(tree.path(node), [0xFFFFFFFF, 7])
        tree.clear_listing(node, 1)
        self.assertFalse(tree.has_listing(node, 1))
        with self.assertRaises(ValueError):
            tree.expand(node, 1, 16)

    def test_model_cache_eviction_and_reconstruction(self):
        model, _ = fixture()
        measure = model.measures[0]
        with patch.object(service, "_CACHED", 1):
            first = model.expand(0, [0], 16)
            self.assertEqual(model.expand(0, [0], 32)[0].label, first[0].label)
            self.assertEqual(len(measure.calls), 1)
            model.expand(0, [1], 16)
            self.assertFalse(model.tree.has_listing(model.tree.node([0]), 0))
            self.assertEqual(model.expand(0, [0], 16), first)
            self.assertEqual(len(measure.calls), 3)
        self.assertEqual(measure.calls[0], (99, 0))

    def test_terminal_zero_horizon_and_bad_labels(self):
        model, _ = fixture()
        self.assertEqual(model.expand(0, [], 0), [])
        self.assertEqual(model.expand(0, [0] * 5, 40), [])
        self.assertEqual(model.measures[0].calls, [])
        for path in [[5], [-1]]:
            with self.assertRaises(service.BadLabel):
                model.expand(0, path, 16)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.model, application = fixture()
        self.client = TestClient(application)

    def expand(self, body):
        return self.client.post(
            "/0/natlas.v1.Natlas/Expand",
            content=body,
            headers={
                "content-type": "application/connect+proto",
                "origin": "http://localhost:5173",
            },
        )

    def test_expansion_streams_edges_directly_by_path(self):
        request = pb.ExpandRequest(path=[0, 1], horizon=16)
        response = self.expand(service._envelope(0, request.SerializeToString()))
        self.assertEqual(response.status_code, 200)
        frames = messages(response.content)
        self.assertEqual(
            [pb.Edge.FromString(data).label for flags, data in frames if not flags],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(json.loads(frames[-1][1]), {})
        self.assertEqual(self.model.measures[0].calls, [(99, 0, 1)])
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_search_and_manifest_have_no_start_path(self):
        response = self.client.post(
            "/1/natlas.v1.Natlas/Search",
            content=pb.SearchRequest(text="abc").SerializeToString(),
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            list(pb.SearchResponse.FromString(response.content).path), [0, 1, 2]
        )
        self.assertNotIn("start", self.client.get("/manifest.json").json()[0])
        self.assertEqual(self.model.measures[0].calls, [])

    def test_invalid_requests_return_protocol_errors(self):
        for body in [b"", b"\x00\x00\x00\x00\x04\x80", service._envelope(0, b"\x80")]:
            frames = messages(self.expand(body).content)
            self.assertEqual(
                json.loads(frames[-1][1])["error"]["code"], "invalid_argument"
            )
        response = self.client.post(
            "/0/natlas.v1.Natlas/Search",
            content=b"\x80",
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 400)


class MigrationTests(unittest.TestCase):
    def test_old_handles_convert_even_out_of_order(self):
        spec = importlib.util.spec_from_file_location(
            "migration", ROOT / "scripts/migrate-snapshots.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        old = migration.legacy_schema()()
        child = old.expansions.add(node=20)
        child.request.node.parent = 10
        child.request.node.label = 0xFFFFFFFF
        child.request.horizon = 8
        child.edges.add(label=3, odds=-123)
        old.expansions.add(node=10).request.horizon = 8
        converted = migration.convert(old.SerializeToString())
        current = pb.Snapshot.FromString(converted)
        self.assertEqual(current.format_version, 2)
        self.assertEqual(list(current.expansions[0].request.path), [0xFFFFFFFF])
        self.assertEqual(current.expansions[0].edges[0].odds, -123)
        self.assertIsNone(migration.convert(converted))


if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn

        uvicorn.run(
            fixture()[1],
            host="127.0.0.1",
            port=int(os.environ.get("PORT", "8423")),
            access_log=False,
        )
    else:
        unittest.main()
