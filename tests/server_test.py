# /// script
# requires-python = ">=3.12"
# dependencies = ["natlas-core", "numpy>=2", "protobuf>=7.36.1", "starlette>=0.40", "httpx2>=2.0", "uvicorn>=0.30"]
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
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

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


class Measure:
    name = "fixture"

    def __init__(self, weights=(0.6, 0.25, 0.1, 0.049, 0.001)):
        self.weights = np.array(weights, dtype=np.float64)
        # Every prefix whose distribution was computed, in order.
        self.calls = []
        # Every walk begun, as (tokens, start).
        self.walks = []

    def walk(self, tokens, start):
        self.walks.append((tuple(tokens), start))
        for end in range(start, len(tokens) + 1):
            self.calls.append(tuple(tokens[:end]))
            with np.errstate(divide="ignore"):
                yield np.log(self.weights)


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


WHOLE = (0.0, -math.inf)


def runs(model, node, path, horizon=16, measure=0, window=WHOLE):
    return list(model.expand(measure, node, path, *window, horizon, threading.Event()))


def weights(run):
    """Each child's weight, from the run's edges."""
    below = np.exp(run.ln_below)
    return list(below[:-1] - below[1:])


class CoreTests(unittest.TestCase):
    def test_tree_names_nodes_by_the_protocol_hash_across_threads(self):
        tree = Tree(1)
        nodes = tree.walk(0, [0xFFFFFFFF, 7])
        self.assertEqual(nodes, [0, node_id([0xFFFFFFFF]), node_id([0xFFFFFFFF, 7])])
        self.assertEqual(tree.walk(nodes[2], []), nodes[2:])
        with ThreadPoolExecutor() as pool:
            found = pool.submit(tree.walk, node_id([0xFFFFFFFF]), [7]).result()
        self.assertEqual(found, nodes[1:])
        self.assertEqual(tree.path(nodes[2]), [0xFFFFFFFF, 7])
        with self.assertRaises(KeyError):
            tree.walk(node_id([1, 2]), [])

    def test_a_walk_lists_what_the_core_holds_and_asks_for_what_it_lacks(self):
        tree = Tree(1)
        walk = tree.start(0, [], 0.0, -math.inf, 2)
        self.assertEqual(tree.advance(walk, 0), ("missing", 0))
        with np.errstate(divide="ignore"):
            tree.set(0, 0, np.log([0.2, 0.5, 0.0, 0.3]))
        kind, node, rank, branches, ln_below = tree.advance(walk, 0)
        self.assertEqual((kind, node, rank, branches), ("run", 0, 0, [1, 3]))
        self.assertAlmostEqual(math.exp(ln_below[-1]), 0.2, places=12)
        self.assertIsNone(tree.advance(walk, 0))
        with self.assertRaises(ValueError):
            tree.start(0, [], -1.0, -0.5, 1)


class ModelTests(unittest.TestCase):
    def test_a_path_is_placed_branch_by_branch_from_one_pass_of_the_model(self):
        model, _ = fixture()
        walked = runs(model, 0, [0, 1, 2], horizon=1)
        self.assertEqual([list(run.branches) for run in walked], [[0], [1], [2], [0]])
        self.assertEqual([run.rank for run in walked], [0, 1, 2, 0])
        measure = model.measures[0]
        self.assertEqual(measure.walks, [((99, 0, 1, 2), 1)])
        self.assertEqual(len(measure.calls), 4)
        # The nodes along it are kept: naming the last by its parent
        # computes nothing.
        again = runs(model, node_id([0, 1]), [2], horizon=1)
        self.assertEqual(again, walked[2:])
        self.assertEqual(len(measure.calls), 4)

    def test_a_window_lists_down_to_its_horizon_or_descends_into_one_child(self):
        model, _ = fixture()
        (root,) = runs(model, 0, [], horizon=4)
        self.assertEqual(list(root.branches), [0, 1, 2])
        self.assertAlmostEqual(sum(weights(root)), 0.95, places=12)
        # (0.5, 0.9] of the root lies inside branch 0's (0.4, 1].
        descent = runs(model, 0, [], horizon=4, window=(math.log(0.9), math.log(0.5)))
        self.assertEqual([list(run.branches) for run in descent[:1]], [[0]])
        self.assertEqual(model.measures[0].calls[-1], (99, 0))

    def test_eviction_forgets_the_oldest_and_recomputes_it_alike(self):
        model, _ = fixture()
        measure = model.measures[0]
        with patch.object(service, "_CACHED", 1):
            first = runs(model, 0, [0])
            runs(model, 0, [1])
            self.assertEqual(runs(model, 0, [0]), first)
        self.assertEqual(measure.calls.count((99,)), 3)

    def test_ended_and_weightless_branches(self):
        model, _ = fixture()
        ended = runs(model, 0, [0] * 5, 40)
        self.assertEqual(len(ended), 6)
        self.assertEqual(ended[-1], pb.Run(rank=0, ln_below=[-math.inf]))
        # Branch 5 is past the vocabulary: nothing places it.
        self.assertEqual(len(runs(model, 0, [5], 40)), 0)
        with self.assertRaises(KeyError):
            runs(model, node_id([3, 3]), [])

    def test_a_canceled_walk_stops_at_the_next_node(self):
        model, _ = fixture()
        canceled = threading.Event()
        walk = model.expand(0, 0, [0, 1, 2], *WHOLE, 16, canceled)
        next(walk)
        canceled.set()
        self.assertEqual(list(walk), [])
        self.assertFalse(model.lock.locked())


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

    def runs(self, node, path, horizon=16, window=WHOLE):
        request = pb.ExpandRequest(
            node=node, path=path, ln_top=window[0], ln_bottom=window[1], horizon=horizon
        )
        response = self.expand(service._envelope(0, request.SerializeToString()))
        self.assertEqual(response.status_code, 200)
        frames = messages(response.content)
        return [
            pb.Run.FromString(data) for flags, data in frames if not flags
        ], json.loads(frames[-1][1])

    def test_a_stream_holds_one_run_per_node_down_the_path(self):
        runs, end = self.runs(0, [0, 1], horizon=1)
        self.assertEqual(end, {})
        self.assertEqual([list(run.branches) for run in runs], [[0], [1], [0]])
        again, _ = self.runs(node_id([0]), [1], horizon=1)
        self.assertEqual(again, runs[1:])
        self.assertEqual(self.model.measures[0].calls, [(99,), (99, 0), (99, 0, 1)])
        response = self.expand(
            service._envelope(0, pb.ExpandRequest(node=0).SerializeToString())
        )
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_an_unlearned_id_is_not_found(self):
        runs, end = self.runs(node_id([2]), [3])
        self.assertEqual(runs, [])
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
        _, end = self.runs(node_id([0, 1]), [2])
        self.assertEqual(end, {})
        self.assertEqual(self.model.measures[0].calls, [(99, 0, 1), (99, 0, 1, 2)])
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
        empty = pb.ExpandRequest(node=0).SerializeToString()
        for body in [b"", b"\x00\x00\x00\x00\x04\x80", service._envelope(0, b"\x80"), service._envelope(0, empty)]:
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
    def test_listing_recordings_become_runs_once(self):
        spec = importlib.util.spec_from_file_location(
            "migration", ROOT / "scripts/migrate-snapshots.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        Snapshot = migration.listing_schema()
        old = Snapshot()
        expansion = old.expansions.add(path=[0xFFFFFFFF], horizon=8)
        expansion.listing.branches.extend([3, 1])
        expansion.listing.ln_weights.extend([math.log(0.5), math.log(0.25)])
        expansion.listing.ln_leftover = math.log(0.25)
        converted = migration.convert(old.SerializeToString(), Snapshot)
        (recording,) = pb.Snapshot.FromString(converted).recordings
        self.assertEqual(list(recording.path), [0xFFFFFFFF])
        self.assertEqual((recording.run.rank, list(recording.run.branches)), (0, [3, 1]))
        self.assertEqual(
            [round(math.exp(edge), 12) for edge in recording.run.ln_below], [1.0, 0.5, 0.25]
        )
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
