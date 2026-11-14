# /// script
# requires-python = ">=3.12"
# dependencies = ["natlas-core", "numpy>=2", "protobuf>=7.36.1", "starlette>=0.40", "httpx2>=2.0", "uvicorn>=0.30"]
# [tool.uv.sources]
# natlas-core = { path = "../core" }
# ///
"""Native core, scheduler and session transport tests without loading an ML model.

--serve starts the same deterministic fixture for browser verification.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import service  # noqa: E402
from natlas import natlas_pb2 as pb  # noqa: E402
from natlas_codec import labels_json  # noqa: E402
from errors import TooLong  # noqa: E402
from natlas_core import Session, Tree  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

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

    def __init__(self, weights=(0.6, 0.25, 0.1, 0.049, 0.001), positions=1, longest=None):
        self.weights = np.array(weights, dtype=np.float64)
        self.positions = positions
        # Prefixes longer than this are too long for the model's context.
        self.longest = longest
        # Every prefix whose distribution was computed, in order.
        self.calls = []
        # Every walk begun, as (tokens, start).
        self.walks = []
        # Prefixes whose computation waits on a gate, and a flag raised when
        # one is reached.
        self.gates = {}
        self.reached = threading.Event()

    def walk(self, tokens, start):
        self.walks.append((tuple(tokens), start))
        for end in range(start, len(tokens) + 1):
            prefix = tuple(tokens[:end])
            if self.longest is not None and self.longest < len(prefix):
                raise TooLong(f"{len(prefix)} tokens, but the context holds {self.longest}")
            if prefix in self.gates:
                self.reached.set()
                self.gates[prefix].wait(5)
            self.calls.append(prefix)
            with np.errstate(divide="ignore"):
                yield np.log(self.weights)


def fixture(**measure):
    measures = [Measure(**measure), Measure((0.25, 0.6, 0.1, 0.049, 0.001))]
    model = service.Model((99,), lambda path: len(path) >= 5, 5, measures)
    manifest = [
        {
            "name": "fixture",
            "title": "Integer sequence fixture",
            "labels": "labels.json",
            "measures": [{"name": f"measure {index}", "url": f"{index}/"} for index in range(2)],
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


class Received:
    """A client of the model, keeping what it is sent."""

    def __init__(self, model):
        self.updates = []
        self.client = model.open(self.send).result(5)

    def send(self, frame):
        self.updates.extend(pb.Updates.FromString(frame).updates)

    def kinds(self, kind):
        return [getattr(update, kind) for update in self.updates if update.WhichOneof("kind") == kind]

    def answered(self, node, measure=0):
        """The branches answered for a node, in order."""
        return [
            branch
            for answer in self.kinds("answer")
            if (answer.node, answer.measure) == (node, measure)
            for run in answer.children.runs
            for branch in run.branches
        ]


def demand(p=0, q=0, view=None, path=None):
    """A demand: the view as (node, rows, right) of the node's whole mass, the
    path as (node, branches)."""
    message = pb.Demand(p=p, q=q)
    if view is not None:
        node, rows, right = view
        message.view.CopyFrom(pb.View(node=node, top=b"", ln_height=0.0, rows=rows, right=right))
    if path is not None:
        node, branches = path
        message.path.CopyFrom(pb.Path(node=node, branches=branches))
    return message


def settle(model):
    """Wait until the scheduler has sent everything and has nothing to compute."""

    def idle():
        for client in model.clients:
            model._step(client)  # noqa: SLF001
        return not model.passes and not any(client.session.blocked() for client in model.clients)

    deadline = time.monotonic() + 5
    while not model.call(idle).result(5):
        if deadline < time.monotonic():
            raise AssertionError("the scheduler did not settle")
        time.sleep(0.001)


def queued(model, client):
    """Wait until a pass for ``client`` is submitted."""
    deadline = time.monotonic() + 5
    while not model.call(lambda: any(job.client is client for job in model.passes.values())).result(5):
        if deadline < time.monotonic():
            raise AssertionError("no pass was submitted for the client")
        time.sleep(0.001)


def held(model):
    """Hold the scheduler until the returned gate is set: what is posted
    meanwhile is taken at once, in order."""
    gate = threading.Event()
    entered = threading.Event()

    def hold():
        entered.set()
        gate.wait(5)

    model.call(hold)
    if not entered.wait(5):
        raise AssertionError("the scheduler did not reach the gate")
    return gate


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

    def test_a_session_answers_what_the_tree_holds_and_waits_on_what_it_lacks(self):
        tree = Tree(1)
        session = Session()
        session.demand(tree, 0, 0, view=(0, b"", 0.0, 10.0, 0.1))
        self.assertEqual(session.step(tree), [])
        self.assertEqual(session.blocked(), [(0, 0, 10.0, [])])
        with np.errstate(divide="ignore"):
            tree.set(0, 0, np.log([0.2, 0.5, 0.0, 0.3]))
        ((kind, measure, node, parent, branch, runs, count),) = session.step(tree)
        self.assertEqual((kind, measure, node, parent, branch, count), ("answer", 0, 0, None, None, 3))
        self.assertEqual([branches for _, branches, _ in runs], [[1, 3, 0]])
        self.assertEqual(session.blocked(), [])
        with self.assertRaises(ValueError):
            session.demand(tree, 0, 1)


class ModelTests(unittest.TestCase):
    def test_a_view_is_computed_from_the_root_down_the_most_rows_first(self):
        model, _ = fixture()
        client = Received(model)
        model.demand(client.client, demand(view=(0, 10.0, 2.0))).result(5)
        settle(model)
        calls = model.measures[0].calls
        # Ten rows: 6 on branch 0, then 3.6 on its own branch 0, 2.5 on
        # branch 1, 2.16 below 0, 0, and 1.5 each on two more; what starts
        # past two nats, or is worth less than a row, is not computed. Each
        # pass is chosen while the one before runs, from what is known then:
        # (1) goes while (0) runs, before (0) tells of (0, 0).
        self.assertEqual(calls[:3], [(99,), (99, 0), (99, 1)])
        self.assertEqual(set(calls[3:]), {(99, 0, 0), (99, 0, 0, 0), (99, 0, 1), (99, 1, 0)})
        self.assertEqual(len(calls), 7)
        self.assertEqual(client.answered(0)[:3], [0, 1, 2])
        (answer,) = [answer for answer in client.kinds("answer") if answer.node == node_id([0, 0])]
        self.assertEqual((answer.parent, answer.branch), (node_id([0]), 0))

    def test_clients_take_turns_pass_by_pass(self):
        model, _ = fixture()
        first, second = Received(model), Received(model)
        gate = held(model)
        one = model.demand(first.client, demand(path=(0, [1, 1, 1])))
        two = model.demand(second.client, demand(path=(0, [2, 2, 2])))
        gate.set()
        one.result(5)
        two.result(5)
        settle(model)
        self.assertEqual(model.measures[0].calls, [(99,), (99, 2), (99, 1), (99, 2, 2), (99, 1, 1)])
        self.assertEqual([list(path.branches) for path in first.kinds("placed")], [[1, 1, 1]])
        self.assertEqual([list(path.branches) for path in second.kinds("placed")], [[2, 2, 2]])

    def test_a_pass_computes_as_much_of_a_path_as_the_model_reads_at_once(self):
        model, _ = fixture(positions=3)
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0, 1, 2, 3]))).result(5)
        settle(model)
        measure = model.measures[0]
        self.assertEqual(measure.walks, [((99, 0, 1), 1), ((99, 0, 1, 2, 3), 4)])
        self.assertEqual(client.answered(node_id([0, 1, 2])), [3])
        self.assertEqual(len(client.kinds("placed")), 1)

    def test_a_new_demand_drops_what_the_old_one_waited_on(self):
        model, _ = fixture()
        measure = model.measures[0]
        gate = measure.gates[(99, 1)] = threading.Event()
        client = Received(model)
        model.demand(client.client, demand(path=(0, [1, 1, 1])))
        self.assertTrue(measure.reached.wait(5))
        # While (1) is computed, the client looks elsewhere, and the
        # scheduler takes it at once.
        model.demand(client.client, demand(path=(0, [2]))).result(5)
        gate.set()
        settle(model)
        self.assertEqual(measure.calls, [(99,), (99, 1)])
        self.assertEqual([list(path.branches) for path in client.kinds("placed")], [[2]])

    def test_a_burst_of_demands_computes_only_what_the_latest_wants(self):
        model, _ = fixture()
        client = Received(model)
        gate = held(model)
        # Views under the other measure, each superseded before it is read.
        for rows in range(1, 100):
            model.demand(client.client, demand(p=1, q=1, view=(0, float(rows), 1.0)))
        latest = model.demand(client.client, demand(path=(0, [2])))
        gate.set()
        latest.result(5)
        settle(model)
        self.assertEqual(model.measures[1].calls, [])
        self.assertEqual(model.measures[0].calls, [(99,)])
        self.assertEqual([list(path.branches) for path in client.kinds("placed")], [[2]])

    def test_the_next_pass_is_queued_while_one_runs_and_withdrawn_on_close(self):
        model, _ = fixture()
        measure = model.measures[0]
        gate = measure.gates[(99,)] = threading.Event()
        first, second = Received(model), Received(model)
        model.learn([2]).result(5)
        model.demand(first.client, demand(path=(0, [1])))
        self.assertTrue(measure.reached.wait(5))
        model.demand(second.client, demand(path=(node_id([2]), [0])))
        queued(model, second.client)
        model.close(second.client).result(5)
        gate.set()
        settle(model)
        self.assertEqual(measure.calls, [(99,)])
        self.assertEqual(second.updates, [])

    def test_a_distribution_on_its_way_is_not_computed_again(self):
        model, _ = fixture()
        measure = model.measures[0]
        gate = measure.gates[(99,)] = threading.Event()
        first, second = Received(model), Received(model)
        model.demand(first.client, demand(path=(0, [1])))
        self.assertTrue(measure.reached.wait(5))
        model.demand(second.client, demand(path=(0, [2]))).result(5)
        gate.set()
        settle(model)
        self.assertEqual(measure.calls, [(99,)])
        self.assertEqual([list(path.branches) for path in first.kinds("placed")], [[1]])
        self.assertEqual([list(path.branches) for path in second.kinds("placed")], [[2]])

    def test_ended_sequences_and_branches_past_the_vocabulary_have_no_children(self):
        model, _ = fixture()
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0] * 5))).result(5)
        settle(model)
        model.learn([7]).result(5)
        for node in (node_id([0] * 5), node_id([7])):
            model.demand(client.client, demand(view=(node, 10.0, 1.0))).result(5)
            settle(model)
            (answer,) = [answer for answer in client.kinds("answer") if answer.node == node]
            self.assertEqual((list(answer.children.runs), answer.children.count), ([], 0))
        self.assertTrue(all(len(call) <= 5 for call in model.measures[0].calls))
        self.assertNotIn((99, 7), model.measures[0].calls)

    def test_a_distribution_that_cannot_be_computed_fails_what_waits_on_it(self):
        model, _ = fixture(longest=3)
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0, 0, 0, 0]))).result(5)
        settle(model)
        (failed,) = client.kinds("failed")
        self.assertEqual((failed.measure, failed.node), (0, node_id([0, 0, 0])))
        self.assertIn("context", failed.message)

    def test_eviction_forgets_the_oldest_and_recomputes_it_alike(self):
        model, _ = fixture()
        model.cache = 1
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0, 1]))).result(5)
        settle(model)
        model.demand(client.client, demand(path=(0, [0, 1, 2]))).result(5)
        settle(model)
        self.assertEqual(model.measures[0].calls.count((99,)), 2)
        self.assertEqual(len(client.kinds("placed")), 2)

    def test_a_node_the_server_never_learned_is_said_unknown(self):
        model, _ = fixture()
        client = Received(model)
        model.demand(client.client, demand(view=(node_id([3, 3]), 10.0, 1.0))).result(5)
        settle(model)
        self.assertEqual(client.kinds("unknown"), [node_id([3, 3])])
        self.assertEqual(model.measures[0].calls, [])


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.model, application = fixture()
        self.client = TestClient(application)

    def updates_until_placed(self, socket):
        updates = []
        while not any(update.WhichOneof("kind") == "placed" for update in updates):
            updates.extend(pb.Updates.FromString(socket.receive_bytes()).updates)
        return updates

    def test_a_session_answers_its_demand_over_one_socket(self):
        with self.client.websocket_connect("/session") as socket:
            socket.send_bytes(demand(path=(0, [0, 1])).SerializeToString())
            updates = self.updates_until_placed(socket)
        answers = [(update.answer.node, [b for run in update.answer.children.runs for b in run.branches]) for update in updates if update.WhichOneof("kind") == "answer"]
        self.assertEqual(answers, [(0, [0]), (node_id([0]), [1])])
        self.assertEqual(self.client.get("/manifest.json").json()[0]["name"], "fixture")

    def test_a_demand_the_server_cannot_take_closes_the_session(self):
        with self.client.websocket_connect("/session") as socket:
            socket.send_bytes(pb.Demand(p=0, q=9).SerializeToString())
            with self.assertRaises(WebSocketDisconnect) as closed:
                socket.receive_bytes()
        self.assertEqual(closed.exception.code, 1007)

    def test_search_teaches_the_server_the_ids_along_the_text(self):
        response = self.client.post(
            "/1/natlas.Natlas/Search",
            content=pb.SearchRequest(text="abc").SerializeToString(),
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(pb.SearchResponse.FromString(response.content).path), [0, 1, 2])
        self.assertEqual(self.model.measures[0].calls, [])
        with self.client.websocket_connect("/session") as socket:
            socket.send_bytes(demand(path=(node_id([0, 1]), [2])).SerializeToString())
            self.updates_until_placed(socket)
        self.assertEqual(self.model.measures[0].calls, [(99, 0, 1)])
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
