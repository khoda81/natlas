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
from natlas import natlas_pb2 as pb  # noqa: E402
from natlas_codec import labels_json  # noqa: E402
from natlas_core import Tree  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

MASK = (1 << 64) - 1


def child_hash(parent, branch):
    """The node id the protocol defines, written out independently of the core."""
    z = (parent + 0x9E3779B97F4A7C15 * (branch + 1)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def node_id(path):
    node = 0
    for branch in path:
        node = child_hash(node, branch)
    return node


class Values:
    def __init__(self, values):
        self.values = values

    def sort(self, descending=True, stable=True):
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
        return Values([math.log(w) if w else -math.inf for w in self.weights])


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


def branches(listings):
    return [branch for listing in listings for branch in listing.branches]


class CoreTests(unittest.TestCase):
    def test_tree_names_nodes_by_the_protocol_hash_across_threads(self):
        tree = Tree(2)
        node = tree.node(0, [0xFFFFFFFF, 7])
        self.assertEqual(tree.node(0, [0xFFFFFFFF, 7]), node)
        self.assertEqual(tree.node(node_id([0xFFFFFFFF, 7]), []), node)
        self.assertEqual(tree.node(node_id([0xFFFFFFFF]), [7]), node)
        with self.assertRaises(KeyError):
            tree.node(node_id([1, 2]), [])
        tree.set_distribution(node, 1, [3, 8], [math.log(0.75), math.log(0.25)])
        with ThreadPoolExecutor() as pool:
            batches = pool.submit(tree.batches, node, 1, 16, 1).result()
        self.assertEqual(
            batches,
            [
                ([3], [math.log(0.75)], math.log(0.25)),
                ([8], [math.log(0.25)], -math.inf),
            ],
        )
        self.assertEqual(
            tree.batches(node, 1, 1, 8), [([3], [math.log(0.75)], math.log(0.25))]
        )
        self.assertEqual(tree.path(node), [0xFFFFFFFF, 7])
        tree.clear_listing(node, 1)
        self.assertFalse(tree.has_listing(node, 1))
        with self.assertRaises(ValueError):
            tree.batches(node, 1, 16, 8)

    def test_model_cache_eviction_and_reconstruction(self):
        model, _ = fixture()
        measure = model.measures[0]
        with patch.object(service, "_CACHED", 1):
            first = model.expand(0, 0, [0], 16)
            self.assertEqual(
                branches(model.expand(0, node_id([0]), [], 32)), branches(first)
            )
            self.assertEqual(len(measure.calls), 1)
            model.expand(0, 0, [1], 16)
            self.assertFalse(model.tree.has_listing(model.tree.node(0, [0]), 0))
            self.assertEqual(model.expand(0, 0, [0], 16), first)
            self.assertEqual(len(measure.calls), 3)
        self.assertEqual(measure.calls[0], (99, 0))

    def test_terminal_zero_horizon_and_zero_weight_branches(self):
        model, _ = fixture()
        self.assertEqual(model.expand(0, 0, [], 0), [pb.Listing(ln_leftover=0.0)])
        for path in [[0] * 5, [5], [0, 0xFFFFFFFF]]:
            self.assertEqual(
                model.expand(0, 0, path, 40), [pb.Listing(ln_leftover=-math.inf)]
            )
        self.assertEqual(model.measures[0].calls, [])
        with self.assertRaises(KeyError):
            model.expand(0, node_id([3, 3]), [], 16)

    def test_weights_are_exact_and_cut_at_the_horizon(self):
        model, _ = fixture()
        (listing,) = model.expand(0, 0, [], 4)
        self.assertEqual(list(listing.branches), [0, 1, 2])
        self.assertEqual(
            list(listing.ln_weights), [math.log(w) for w in (0.6, 0.25, 0.1)]
        )
        self.assertAlmostEqual(math.exp(listing.ln_leftover), 0.05, places=12)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.model, application = fixture()
        self.client = TestClient(application)

    def expand(self, body):
        return self.client.post(
            "/0/natlas.Natlas/Expand",
            content=body,
            headers={
                "content-type": "application/connect+proto",
                "origin": "http://localhost:5173",
            },
        )

    def listings(self, node, path, horizon=16):
        request = pb.ExpandRequest(node=node, path=path, horizon=horizon)
        response = self.expand(service._envelope(0, request.SerializeToString()))
        self.assertEqual(response.status_code, 200)
        frames = messages(response.content)
        return [
            pb.Listing.FromString(data) for flags, data in frames if not flags
        ], json.loads(frames[-1][1])

    def test_nodes_are_named_by_a_known_id_and_one_branch(self):
        listings, end = self.listings(0, [0, 1])
        self.assertEqual(end, {})
        self.assertEqual(branches(listings), [0, 1, 2, 3, 4])
        self.assertEqual(listings[-1].ln_leftover, -math.inf)
        again, _ = self.listings(node_id([0]), [1])
        self.assertEqual(again, listings)
        self.assertEqual(self.model.measures[0].calls, [(99, 0, 1)])
        response = self.expand(
            service._envelope(0, pb.ExpandRequest(node=0).SerializeToString())
        )
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_an_unlearned_id_is_not_found(self):
        listings, end = self.listings(node_id([2]), [3])
        self.assertEqual(listings, [])
        self.assertEqual(end["error"]["code"], "not_found")

    def test_search_teaches_the_server_the_ids_along_the_text(self):
        response = self.client.post(
            "/1/natlas.Natlas/Search",
            content=pb.SearchRequest(text="abc").SerializeToString(),
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            list(pb.SearchResponse.FromString(response.content).path), [0, 1, 2]
        )
        self.assertEqual(self.model.measures[0].calls, [])
        listings, end = self.listings(node_id([0, 1]), [2])
        self.assertEqual(end, {})
        self.assertEqual(self.model.measures[0].calls, [(99, 0, 1, 2)])
        self.assertNotIn("start", self.client.get("/manifest.json").json()[0])

    def test_search_answers_while_a_forward_pass_runs(self):
        def search():
            return self.client.post(
                "/0/natlas.Natlas/Search",
                content=pb.SearchRequest(text="ab").SerializeToString(),
                headers={"content-type": "application/proto"},
            )

        with self.model.lock, ThreadPoolExecutor() as pool:
            response = pool.submit(search).result(timeout=5)
        self.assertEqual(response.status_code, 200)

    def test_invalid_requests_return_protocol_errors(self):
        for body in [b"", b"\x00\x00\x00\x00\x04\x80", service._envelope(0, b"\x80")]:
            frames = messages(self.expand(body).content)
            self.assertEqual(
                json.loads(frames[-1][1])["error"]["code"], "invalid_argument"
            )
        response = self.client.post(
            "/0/natlas.Natlas/Search",
            content=b"\x80",
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 400)


class MigrationTests(unittest.TestCase):
    def test_odds_recordings_convert_to_exact_listings_once(self):
        spec = importlib.util.spec_from_file_location(
            "migration", ROOT / "scripts/migrate-snapshots.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        Snapshot = migration.odds_schema()
        old = Snapshot(format_version=2)
        expansion = old.expansions.add()
        expansion.request.path.extend([0xFFFFFFFF])
        expansion.request.horizon = 8
        # Half the interval, then a tie with it: nothing is left.
        expansion.edges.add(label=3, odds=0)
        expansion.edges.add(label=1, odds=migration.SATURATED)
        converted = migration.convert(old.SerializeToString(), Snapshot)
        (current,) = pb.Snapshot.FromString(converted).expansions
        self.assertEqual(list(current.path), [0xFFFFFFFF])
        self.assertEqual(current.horizon, 8)
        self.assertEqual(list(current.listing.branches), [3, 1])
        self.assertEqual(list(current.listing.ln_weights), [math.log(0.5)] * 2)
        self.assertEqual(current.listing.ln_leftover, -math.inf)
        self.assertIsNone(migration.convert(converted, Snapshot))


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
