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


def request(node, path=None, points=None, branches=None):
    """Expand down ``path``, or at ``points``, or along ``branches``."""
    if points is not None:
        read = {"points": pb.Points(ln_below=points)}
    elif branches is not None:
        read = {"branches": pb.Branches(branches=branches)}
    else:
        read = {"path": pb.Path(branches=path or [])}
    return pb.ExpandRequest(node=node, **read)


def answers(model, node, path=None, points=None, branches=None, measure=0):
    return list(model.expand(measure, request(node, path, points, branches), threading.Event()))


def placed(children):
    """The branches one message places, run by run."""
    return [list(run.branches) for run in children.runs]


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

    def test_a_walk_places_what_the_core_holds_and_asks_for_what_it_lacks(self):
        tree = Tree(1)
        walk = tree.start_points(0, [math.log(0.9), math.log(0.4)])
        self.assertEqual(tree.advance(walk, 0), ("missing", 0))
        with np.errstate(divide="ignore"):
            tree.set(0, 0, np.log([0.2, 0.5, 0.0, 0.3]))
        kind, node, runs, count = tree.advance(walk, 0)
        self.assertEqual((kind, node, count), ("children", 0, 3))
        ((rank, branches, ln_below),) = runs
        self.assertEqual((rank, branches), (0, [1, 3]))
        self.assertAlmostEqual(math.exp(ln_below[-1]), 0.2, places=12)
        self.assertIsNone(tree.advance(walk, 0))
        with self.assertRaises(ValueError):
            tree.start_points(0, [0.5])


class ModelTests(unittest.TestCase):
    def test_a_path_is_placed_branch_by_branch_from_one_pass_of_the_model(self):
        model, _ = fixture()
        walked = answers(model, 0, [0, 1, 2])
        self.assertEqual([placed(children) for children in walked], [[[0]], [[1]], [[2]]])
        self.assertEqual([children.runs[0].rank for children in walked], [0, 1, 2])
        measure = model.measures[0]
        self.assertEqual(measure.walks, [((99, 0, 1, 2), 1)])
        self.assertEqual(measure.calls, [(99,), (99, 0), (99, 0, 1)])
        # The nodes along it are kept: naming the last by its parent
        # computes nothing.
        self.assertEqual(answers(model, node_id([0, 1]), [2]), walked[2:])
        self.assertEqual(len(measure.calls), 3)

    def test_points_place_the_children_holding_them_or_descend_into_one(self):
        model, _ = fixture()
        # 0.9 lies in branch 0's (0.4, 1], 0.3 in branch 1's (0.15, 0.4].
        (root,) = answers(model, 0, points=[math.log(0.9), math.log(0.3)])
        self.assertEqual((placed(root), root.count), ([[0, 1]], 5))
        # 0.9 and 0.5 both lie in branch 0: the points descend into it,
        # where they fall 0.83 and 0.17 of the way up, in its branches 0 and 1.
        descent = answers(model, 0, points=[math.log(0.9), math.log(0.5)])
        self.assertEqual([placed(children) for children in descent], [[[0]], [[0, 1]]])
        self.assertEqual(model.measures[0].calls[-1], (99, 0))
        (branches,) = answers(model, 0, branches=[4, 1, 7])
        self.assertEqual(placed(branches), [[1], [4]])

    def test_eviction_forgets_the_oldest_and_recomputes_it_alike(self):
        model, _ = fixture()
        measure = model.measures[0]
        with patch.object(service, "_CACHED", 1):
            first = answers(model, 0, [0, 1])
            self.assertEqual(answers(model, 0, [0, 1]), first)
        self.assertEqual(measure.calls.count((99,)), 2)

    def test_ended_and_weightless_branches(self):
        model, _ = fixture()
        self.assertEqual(len(answers(model, 0, [0] * 5)), 5)
        ended = answers(model, node_id([0] * 5), points=[-1.0])
        self.assertEqual(ended, [pb.Children(count=0)])
        # Branch 5 is past the vocabulary: nothing places it.
        self.assertEqual(answers(model, 0, [5]), [])
        with self.assertRaises(KeyError):
            answers(model, node_id([3, 3]), [])

    def test_a_canceled_walk_stops_at_the_next_node(self):
        model, _ = fixture()
        canceled = threading.Event()
        walk = model.expand(0, request(0, [0, 1, 2]), canceled)
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

    def answers(self, node, path):
        response = self.expand(service._envelope(0, request(node, path).SerializeToString()))
        self.assertEqual(response.status_code, 200)
        frames = messages(response.content)
        return [
            pb.Children.FromString(data) for flags, data in frames if not flags
        ], json.loads(frames[-1][1])

    def test_a_stream_holds_one_message_per_node_down_the_path(self):
        answered, end = self.answers(0, [0, 1])
        self.assertEqual(end, {})
        self.assertEqual([placed(children) for children in answered], [[[0]], [[1]]])
        again, _ = self.answers(node_id([0]), [1])
        self.assertEqual(again, answered[1:])
        self.assertEqual(self.model.measures[0].calls, [(99,), (99, 0)])
        response = self.expand(
            service._envelope(0, request(0).SerializeToString())
        )
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_an_unlearned_id_is_not_found(self):
        answered, end = self.answers(node_id([2]), [3])
        self.assertEqual(answered, [])
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
        _, end = self.answers(node_id([0, 1]), [2])
        self.assertEqual(end, {})
        self.assertEqual(self.model.measures[0].calls, [(99, 0, 1)])
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
        untargeted = pb.ExpandRequest(node=0).SerializeToString()
        empty = request(0, points=[0.5]).SerializeToString()
        for body in [
            b"",
            b"\x00\x00\x00\x00\x04\x80",
            service._envelope(0, b"\x80"),
            service._envelope(0, untargeted),
            service._envelope(0, empty),
        ]:
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
    def test_run_recordings_become_children_once(self):
        spec = importlib.util.spec_from_file_location(
            "migration", ROOT / "scripts/migrate-snapshots.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        Snapshot = migration.run_schema()
        old = Snapshot()
        recording = old.recordings.add(path=[0xFFFFFFFF])
        recording.run.branches.extend([3, 1])
        recording.run.ln_below.extend([0.0, math.log(0.5), math.log(0.25)])
        recording.run.children = 7
        converted = migration.convert(old.SerializeToString(), Snapshot)
        (current,) = pb.Snapshot.FromString(converted).recordings
        self.assertEqual(list(current.path), [0xFFFFFFFF])
        (run,) = current.children.runs
        self.assertEqual((run.rank, list(run.branches), current.children.count), (0, [3, 1], 7))
        self.assertEqual(list(run.ln_below), [0.0, math.log(0.5), math.log(0.25)])
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
