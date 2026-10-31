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


def branches(listings):
    return [branch for listing in listings for branch in listing.branches]


def listings(model, node, path, horizon=16, measure=0):
    return list(model.expand(measure, node, path, horizon, threading.Event()))


class CoreTests(unittest.TestCase):
    def test_tree_names_nodes_by_the_protocol_hash_across_threads(self):
        tree = Tree()
        nodes = tree.walk(0, [0xFFFFFFFF, 7])
        self.assertEqual(len(nodes), 3)
        self.assertEqual(tree.walk(0, [0xFFFFFFFF, 7]), nodes)
        self.assertEqual(tree.walk(node_id([0xFFFFFFFF, 7]), []), nodes[2:])
        with ThreadPoolExecutor() as pool:
            found = pool.submit(tree.walk, node_id([0xFFFFFFFF]), [7]).result()
        self.assertEqual(found, nodes[1:])
        self.assertEqual(tree.path(nodes[2]), [0xFFFFFFFF, 7])
        with self.assertRaises(KeyError):
            tree.walk(node_id([1, 2]), [])

    def test_cuts_reach_the_horizon_and_the_next_branch_with_an_exact_rest(self):
        ln_weights = np.log([0.5, 1e-30, 0.3, 0.2 - 1e-30])
        horizon = service.cut(ln_weights, 1, None)
        self.assertEqual(list(horizon.branches), [0])
        self.assertAlmostEqual(math.exp(horizon.ln_leftover), 0.5, places=12)
        self.assertEqual(list(service.cut(ln_weights, 2, None).branches), [0, 2])
        through = service.cut(ln_weights, 1, 1)
        self.assertEqual(list(through.branches), [0, 2, 3, 1])
        self.assertEqual(through.ln_leftover, -math.inf)
        # The rest is summed directly: 1e-30 survives beside the heavy ones.
        rare = service.cut(ln_weights, 3, 3)
        self.assertEqual(list(rare.branches), [0, 2, 3])
        self.assertAlmostEqual(rare.ln_leftover, math.log(1e-30), places=9)
        self.assertEqual(list(service.cut(ln_weights, 0, None).branches), [])
        # Ties with the next branch come along, by increasing branch.
        tied = service.cut(np.log([0.2, 0.4, 0.2, 0.2]), 1, 3)
        self.assertEqual(list(tied.branches), [1, 0, 2, 3])
        # A branch of weight zero or past the vocabulary places nothing more.
        self.assertEqual(
            list(service.cut(np.log([1.0, 1e-300]), 1, 1).branches), [0, 1]
        )
        with np.errstate(divide="ignore"):
            self.assertEqual(list(service.cut(np.log([1.0, 0.0]), 1, 1).branches), [0])
        self.assertEqual(list(service.cut(ln_weights, 1, 9).branches), [0])

    def test_a_walk_lists_every_node_through_the_next_branch_in_one_pass(self):
        model, _ = fixture()
        walked = listings(model, 0, [0, 1, 2], horizon=1)
        self.assertEqual(
            [list(listing.branches) for listing in walked], [[0, 1], [0, 1, 2], [0]]
        )
        measure = model.measures[0]
        self.assertEqual(measure.walks, [((99, 0, 1, 2), 2)])
        self.assertEqual(measure.calls, [(99, 0), (99, 0, 1), (99, 0, 1, 2)])
        # The nodes along it are cached: naming the last by its parent
        # computes nothing.
        again = listings(model, node_id([0, 1]), [2], horizon=1)
        self.assertEqual(again, walked[-1:])
        self.assertEqual(len(measure.calls), 3)
        root = listings(model, 0, [], horizon=4)
        self.assertEqual([list(listing.branches) for listing in root], [[0, 1, 2]])

    def test_model_cache_eviction_and_reconstruction(self):
        model, _ = fixture()
        measure = model.measures[0]
        with patch.object(service, "_CACHED", 1):
            first = listings(model, 0, [0])
            self.assertEqual(
                listings(model, node_id([0]), [], 32)[0].branches, first[0].branches
            )
            self.assertEqual(len(measure.calls), 1)
            listings(model, 0, [1])
            self.assertEqual(listings(model, 0, [0]), first)
            self.assertEqual(len(measure.calls), 3)
        self.assertEqual(measure.calls[0], (99, 0))

    def test_terminal_zero_horizon_and_zero_weight_branches(self):
        model, _ = fixture()
        self.assertEqual(listings(model, 0, [], 0), [pb.Listing(ln_leftover=0.0)])
        ended = listings(model, 0, [0] * 5, 40)
        self.assertEqual(len(ended), 5)
        self.assertEqual(ended[-1], pb.Listing(ln_leftover=-math.inf))
        self.assertEqual(list(ended[-2].branches), [0, 1, 2, 3, 4])
        self.assertEqual(
            listings(model, 0, [5], 40), [pb.Listing(ln_leftover=-math.inf)]
        )
        self.assertEqual(
            listings(model, 0, [5, 0], 40), [pb.Listing(ln_leftover=-math.inf)] * 2
        )
        with self.assertRaises(KeyError):
            listings(model, node_id([3, 3]), [])

    def test_a_canceled_walk_stops_at_the_next_node(self):
        model, _ = fixture()
        canceled = threading.Event()
        walk = model.expand(0, 0, [0, 1, 2], 16, canceled)
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

    def listings(self, node, path, horizon=16):
        request = pb.ExpandRequest(node=node, path=path, horizon=horizon)
        response = self.expand(service._envelope(0, request.SerializeToString()))
        self.assertEqual(response.status_code, 200)
        frames = messages(response.content)
        return [
            pb.Listing.FromString(data) for flags, data in frames if not flags
        ], json.loads(frames[-1][1])

    def test_a_stream_holds_one_listing_per_node_down_the_path(self):
        listings, end = self.listings(0, [0, 1], horizon=1)
        self.assertEqual(end, {})
        self.assertEqual(
            [list(listing.branches) for listing in listings], [[0, 1], [0]]
        )
        again, _ = self.listings(node_id([0]), [1], horizon=1)
        self.assertEqual(again, listings[-1:])
        self.assertEqual(self.model.measures[0].calls, [(99, 0), (99, 0, 1)])
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
