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
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import service  # noqa: E402
from catalog import Catalog, Choice, Number, Text  # noqa: E402
from natlas import natlas_pb2 as pb  # noqa: E402
from natlas_codec import labels_json  # noqa: E402
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
    widths = False
    start = (99,)

    def __init__(self, weights=(0.6, 0.25, 0.1, 0.049, 0.001), width=1, paths=1, longest=None):
        self.weights = np.array(weights, dtype=np.float64)
        self.width, self.paths = width, paths
        # The longest prefix the model's context holds.
        self.longest = longest
        # Every prefix whose distribution was computed, in order.
        self.calls = []
        # Every walk read, as (tokens, start), and the walks of each pass.
        self.walks = []
        self.passes = []
        # Prefixes whose computation waits on a gate, and a flag raised when
        # one is reached.
        self.gates = {}
        self.reached = threading.Event()

    def ends(self, tokens):
        return 5 <= len(tokens)

    def walk(self, walks):
        walks = [(tuple(tokens), start) for tokens, start in walks]
        self.walks.extend(walks)
        self.passes.append(walks)
        assert len(walks) <= self.paths, f"{len(walks)} paths in a pass"
        assert sum(len(tokens) - start + 1 for tokens, start in walks) <= self.width, f"{walks} is wider than a pass"
        computed = []
        for tokens, start in walks:
            rows = []
            for end in range(start, len(tokens) + 1):
                prefix = tokens[:end]
                assert self.longest is None or len(prefix) <= self.longest, f"{prefix} is longer than the context"
                if prefix in self.gates:
                    self.reached.set()
                    self.gates[prefix].wait(5)
                self.calls.append(prefix)
                with np.errstate(divide="ignore"):
                    # Weights up to a constant.
                    rows.append((np.log(self.weights) + 3.0).astype(np.float32))
            computed.append(np.stack(rows))
        return computed


def fixture(**measure):
    """The model, its app, and its models' measures by model: "a", "b" with
    branches 0 and 1 swapped, and "c" alike, which is not loaded; "broken"
    fails to load. A measure's temperature changes nothing."""
    measures = {"a": Measure(**measure), "b": Measure((0.25, 0.6, 0.1, 0.049, 0.001)), "c": Measure()}

    def load(values):
        if values["source"] == "broken":
            raise RuntimeError("no such weights")
        return values["source"]

    catalog = Catalog([Choice("source", ("a", "b", "c", "broken"), "a")], [Number("temperature", 1.0)], load, lambda model, _: measures[model])
    model = service.Model(5, catalog)
    for name in ("source=a", "source=b", "source=broken"):
        model.load(name).result(5)
    wait([model.models[name] for name in ("source=a", "source=b", "source=broken")], 5)
    manifest = [{"name": "fixture", "title": "Integer sequence fixture", "labels": "labels.json", "catalog": catalog.describe()}]
    labels = [b"a", b"b", b"c", b"r", b"\n"]

    def search(text):
        return [b"abcr\n".index(character) for character in text.encode()]

    return model, service.app(
        model,
        manifest,
        {"labels.json": (labels_json(labels).encode(), "application/json")},
        search,
    ), measures


class Received:
    """A client of the model, keeping what it is sent."""

    def __init__(self, model):
        self.updates = []
        self.client = model.open(self.send).result(5)

    def send(self, frame):
        self.updates.extend(pb.Updates.FromString(frame).updates)

    def kinds(self, kind):
        return [getattr(update, kind) for update in self.updates if update.WhichOneof("kind") == kind]

    def answered(self, node, measure="source=a"):
        """The branches answered for a node, in order."""
        return [
            branch
            for answer in self.kinds("answer")
            if (answer.node, answer.measure) == (node, measure)
            for run in answer.children.runs
            for branch in run.branches
        ]


def demand(p="source=a", q="source=a", view=None, path=None):
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


def settle(model, *, until=lambda: True):
    """Wait until the expected result has been sent and the scheduler is idle."""

    def idle():
        for client in model.clients:
            model._step(client)  # noqa: SLF001
        return not model.passes and not any(client.session.blocked() for client in model.clients) and until()

    deadline = time.monotonic() + 5
    while not model.call(idle).result(5):
        if deadline < time.monotonic():
            raise AssertionError("the scheduler did not settle")
        time.sleep(0.001)


def queued(model, client):
    """Wait until a pass for ``client`` is submitted."""
    deadline = time.monotonic() + 5
    while not model.call(lambda: any(client in job.clients for job in model.passes.values())).result(5):
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
        tree = Tree()
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
        tree = Tree()
        session = Session()
        session.demand(tree, 0, 0, view=(0, b"", 0.0, 10.0, 0.1))
        self.assertEqual(session.step(tree), [])
        self.assertEqual(session.blocked(), [(0, 0, 10.0, [])])
        with np.errstate(divide="ignore"):
            row = np.log(np.array([0.2, 0.5, 0.0, 0.3], np.float32))
        tree.set_rows([0], 0, [row], heaviest=True)
        ((kind, measure, node, parent, branch, runs, count),) = session.step(tree)
        self.assertEqual((kind, measure, node, parent, branch, count), ("answer", 0, 0, None, None, 3))
        self.assertEqual([branch for _, branches, _, _ in runs for branch in branches], [1, 3, 0])
        # Each run's top is the mass above it, its weights the children's.
        above = 0.0
        for _, branches, ln_weights, top in runs:
            self.assertAlmostEqual(int.from_bytes(top, "big") / 256 ** len(top), above, delta=256 ** -len(top))
            above += sum(math.exp(ln) for ln in ln_weights)
        self.assertAlmostEqual(above, 1.0)
        # Widths are each child's own ln weight, whatever they sum to.
        widths = np.array([-1.0, -1.0, -3.0], np.float32)
        tree.set([0], 0, [(np.arange(len(widths), dtype=np.uint32), widths.astype(np.float64))])
        tree.set_rows([0], 1, [widths], heaviest=False, normalize=False)
        for measure in (0, 1):
            session = Session()
            session.demand(tree, measure, measure, view=(0, b"", 0.0, 10.0, 0.1))
            ((*_, runs, _),) = session.step(tree)
            self.assertEqual([round(ln, 6) for _, _, ln_weights, _ in runs for ln in ln_weights], [-1.0, -1.0, -3.0])
        # By branch, the same weights in branch order.
        tree.set_rows([0], 0, [row], heaviest=False)
        session = Session()
        session.demand(tree, 0, 0, view=(0, b"", 0.0, 10.0, 0.1))
        ((*_, runs, _),) = session.step(tree)
        self.assertEqual([branch for _, branches, _, _ in runs for branch in branches], [0, 1, 3])
        self.assertEqual(session.blocked(), [])

    def test_inference_rows_keep_masks_ties_and_the_width_of_an_almost_certain_branch(self):
        tree = Tree()
        nodes = tree.walk(0, [7, 9])
        # Take a strided row. Equal logits, including signed zeros, tie by branch.
        row = np.array([0.0, 99.0, -0.0, 99.0, -np.inf, 99.0, 0.5, 99.0, 0.0], np.float32)[::2]
        nearly_certain = np.array([0.0, -50.0], np.float32)
        masked = np.full(2, -np.inf, np.float32)
        sizes = tree.set_rows(nodes, 0, [row, nearly_certain, masked], heaviest=True)
        self.assertEqual(len(sizes), len(nodes))

        session = Session()
        session.demand(tree, 0, 0, view=(nodes[0], b"", 0.0, 10.0, 0.1))
        ((*_, runs, count),) = session.step(tree)
        self.assertEqual(count, 4)
        self.assertEqual([branch for _, branches, _, _ in runs for branch in branches], [3, 0, 1, 4])

        session = Session()
        session.demand(tree, 0, 0, path=(nodes[1], [0]))
        ((*_, runs, count),) = [update for update in session.step(tree) if update[0] == "answer"]
        self.assertEqual(count, 2)
        ((_, branches, [ln_weight], _),) = runs
        self.assertEqual(branches, [0])
        expected = -math.log1p(math.exp(-50.0))
        self.assertAlmostEqual(ln_weight, expected, delta=math.ulp(expected))

        session = Session()
        session.demand(tree, 0, 0, view=(nodes[2], b"", 0.0, 10.0, 0.1))
        ((*_, runs, count),) = session.step(tree)
        self.assertEqual((runs, count), ([], 0))

        with self.assertRaisesRegex(ValueError, "a weight must be a log mass"):
            tree.set_rows(nodes[:2], 1, [row, np.array([np.nan], np.float32)], heaviest=True)
        self.assertFalse(tree.has(nodes[0], 1))


class CatalogTests(unittest.TestCase):
    def test_a_spec_is_a_model_and_optional_measure_parameters_with_defaults_omitted(self):
        models = [Choice("size", ("small", "medium"), "medium"), Choice("device", ("cuda", "cpu"), "cuda")]
        catalog = Catalog(models, [Number("temperature", 1.0), Text("instruments")], dict, dict)
        self.assertEqual(catalog.model("size=small&device=cpu"), {"size": "small", "device": "cpu"})
        self.assertEqual(catalog.parse("size=small&device=cpu"), ("size=small&device=cpu", {"temperature": 1.0, "instruments": ""}))
        self.assertEqual(
            catalog.parse("size=medium|instruments=piano%2Cdrums&temperature=1.6"),
            ("size=medium", {"temperature": 1.6, "instruments": "piano,drums"}),
        )
        for spec in ("", "size=large", "size=small|temperature", "size=small|temperature=inf", "size=small|temperature=1&temperature=2", "size=small|depth=3"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                catalog.parse(spec)
        described = catalog.describe()
        self.assertEqual(described["models"][0], {"name": "size", "kind": "choice", "options": ["small", "medium"], "default": "medium"})
        self.assertEqual(described["parameters"][0], {"name": "temperature", "kind": "number", "default": 1.0})


class ModelTests(unittest.TestCase):
    def test_a_view_is_computed_from_the_root_down_the_most_rows_first(self):
        model, _, measures = fixture()
        client = Received(model)
        model.demand(client.client, demand(view=(0, 10.0, 2.0))).result(5)
        settle(model)
        calls = measures["source=a"].calls
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
        model, _, measures = fixture()
        first, second = Received(model), Received(model)
        gate = held(model)
        one = model.demand(first.client, demand(path=(0, [1, 1, 1])))
        two = model.demand(second.client, demand(path=(0, [2, 2, 2])))
        gate.set()
        one.result(5)
        two.result(5)
        settle(model)
        self.assertEqual(measures["source=a"].calls, [(99,), (99, 2), (99, 1), (99, 2, 2), (99, 1, 1)])
        self.assertEqual([list(path.branches) for path in first.kinds("placed")], [[1, 1, 1]])
        self.assertEqual([list(path.branches) for path in second.kinds("placed")], [[2, 2, 2]])

    def test_a_pass_computes_as_much_of_a_path_as_the_model_reads_at_once(self):
        model, _, measures = fixture(width=3)
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0, 1, 2, 3]))).result(5)
        settle(model)
        measure = measures["source=a"]
        self.assertEqual(measure.walks, [((99, 0, 1), 1), ((99, 0, 1, 2, 3), 4)])
        self.assertEqual(client.answered(node_id([0, 1, 2])), [3])
        self.assertEqual(len(client.kinds("placed")), 1)

    def test_a_pass_computes_what_several_clients_wait_on(self):
        # Each client's walk reads its node and the next along its path.
        one, two = ((99, 1, 0), 2), ((99, 2, 0), 2)
        for paths, passes in ((2, [[one, two]]), (1, [[one], [two]])):
            with self.subTest(paths=paths):
                model, _, measures = fixture(width=4, paths=paths)
                model.learn([1]).result(5)
                model.learn([2]).result(5)
                first, second = Received(model), Received(model)
                gate = held(model)
                demands = [
                    model.demand(first.client, demand(path=(node_id([1]), [0]))),
                    model.demand(second.client, demand(path=(node_id([2]), [0]))),
                ]
                gate.set()
                for demanded in demands:
                    demanded.result(5)
                settle(model)
                self.assertEqual(measures["source=a"].passes, passes)
                for client in (first, second):
                    self.assertEqual([list(path.branches) for path in client.kinds("placed")], [[0]])

    def test_a_view_is_computed_in_passes_as_wide_as_the_measure(self):
        model, _, measures = fixture(width=3, paths=3)
        client = Received(model)
        model.demand(client.client, demand(view=(0, 10.0, 2.0))).result(5)
        settle(model)
        measure = measures["source=a"]
        # What the view computes one at a time, each once.
        self.assertEqual(
            set(measure.calls),
            {(99,), (99, 0), (99, 1), (99, 0, 0), (99, 0, 0, 0), (99, 0, 1), (99, 1, 0)},
        )
        self.assertEqual(len(measure.calls), 7)
        self.assertLess(1, max(map(len, measure.passes)))
        self.assertEqual(client.answered(0)[:3], [0, 1, 2])

    def test_a_new_demand_drops_what_the_old_one_waited_on(self):
        model, _, measures = fixture()
        measure = measures["source=a"]
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
        model, _, measures = fixture()
        client = Received(model)
        gate = held(model)
        # Views under the other measure, each superseded before it is read.
        for rows in range(1, 100):
            model.demand(client.client, demand(p="source=b", q="source=b", view=(0, float(rows), 1.0)))
        latest = model.demand(client.client, demand(path=(0, [2])))
        gate.set()
        latest.result(5)
        settle(model)
        self.assertEqual(measures["b"].calls, [])
        self.assertEqual(measures["source=a"].calls, [(99,)])
        self.assertEqual([list(path.branches) for path in client.kinds("placed")], [[2]])

    def test_the_next_pass_is_queued_while_one_runs_and_withdrawn_on_close(self):
        model, _, measures = fixture()
        measure = measures["source=a"]
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
        model, _, measures = fixture()
        measure = measures["source=a"]
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
        model, _, measures = fixture()
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0] * 5))).result(5)
        settle(model)
        model.learn([7]).result(5)
        for node in (node_id([0] * 5), node_id([7])):
            model.demand(client.client, demand(view=(node, 10.0, 1.0))).result(5)
            settle(model)
            (answer,) = [answer for answer in client.kinds("answer") if answer.node == node]
            self.assertEqual((list(answer.children.runs), answer.children.count), ([], 0))
        self.assertTrue(all(len(call) <= 5 for call in measures["source=a"].calls))
        self.assertNotIn((99, 7), measures["source=a"].calls)

    def test_a_distribution_that_cannot_be_computed_fails_what_waits_on_it(self):
        model, _, measures = fixture(longest=3)
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0, 0, 0, 0]))).result(5)
        settle(model)
        (failed,) = client.kinds("failed")
        self.assertEqual((failed.measure, failed.node), ("source=a", node_id([0, 0, 0])))
        self.assertIn("context", failed.message)

    def test_eviction_forgets_the_oldest_and_recomputes_it_alike(self):
        model, _, measures = fixture()
        model.cache = 1
        client = Received(model)
        model.demand(client.client, demand(path=(0, [0, 1]))).result(5)
        settle(model)
        model.demand(client.client, demand(path=(0, [0, 1, 2]))).result(5)
        settle(model)
        self.assertEqual(measures["source=a"].calls.count((99,)), 2)
        self.assertEqual(len(client.kinds("placed")), 2)

    def test_a_measure_of_a_model_not_loaded_fails_and_is_answered_once_it_loads(self):
        model, _, _ = fixture()
        client = Received(model)
        model.demand(client.client, demand(p="source=c|temperature=2", q="source=a", view=(0, 10.0, 1.0))).result(5)
        settle(model)
        (failed,) = client.kinds("failed")
        self.assertEqual((failed.measure, failed.node), ("c/temperature=2", 0))
        self.assertIn("load c", failed.message)
        # Loaded, it is opened for the client that named it, which says nothing more.
        model.load("source=c").result(5)
        settle(model, until=lambda: bool(client.answered(0, "c/temperature=2")))
        self.assertEqual(client.answered(0, "c/temperature=2")[:2], [0, 1])
        self.assertIn("c/temperature=2", model.held().result(5)["measures"])

    def test_a_model_that_cannot_load_fails_what_waits_on_its_measures(self):
        model, _, _ = fixture()
        client = Received(model)
        model.demand(client.client, demand(p="source=broken", q="source=a", view=(0, 10.0, 1.0))).result(5)
        settle(model)
        (failed,) = client.kinds("failed")
        self.assertEqual((failed.measure, failed.node), ("source=broken", 0))
        self.assertIn("no such weights", failed.message)
        held = model.held().result(5)
        self.assertEqual([loaded["state"] for loaded in held["models"]], ["loaded", "loaded", "failed"])
        self.assertIn("no such weights", held["models"][2]["message"])

    def test_unloading_closes_its_measures_and_what_waits_on_them_fails(self):
        model, _, measures = fixture()
        client = Received(model)
        model.demand(client.client, demand(p="source=b", q="source=a", view=(0, 10.0, 1.0))).result(5)
        settle(model)
        self.assertEqual(client.answered(0, "source=b")[:2], [1, 0])
        self.assertEqual(model.held().result(5)["measures"], ["source=b", "source=a"])
        model.unload("source=b").result(5)
        model.learn([1]).result(5)
        model.demand(client.client, demand(p="source=b", q="source=a", view=(node_id([1]), 10.0, 1.0))).result(5)
        settle(model)
        self.assertEqual({failed.measure for failed in client.kinds("failed")}, {"source=b"})
        self.assertIn("load b", client.kinds("failed")[0].message)
        self.assertEqual(model.held().result(5), {"models": [{"spec": "source=a", "state": "loaded"}, {"spec": "source=broken", "state": "failed", "message": "RuntimeError: no such weights"}], "measures": ["source=a"]})

    def test_unloading_while_a_measure_opens_retires_it(self):
        opening, release = threading.Event(), threading.Event()
        first = Measure()
        second = Measure((0.25, 0.6, 0.1, 0.049, 0.001))
        loads = iter([first, second])

        def open_measure(loaded, _):
            if loaded is first:
                opening.set()
                if not release.wait(5):
                    raise TimeoutError("the test did not release measure opening")
            return loaded

        catalog = Catalog([Choice("source", ("a",), "a")], [], lambda _: next(loads), open_measure)
        model = service.Model(5, catalog)
        model.load("source=a").result(5)
        client = Received(model)
        model.demand(client.client, demand(view=(0, 10.0, 1.0))).result(5)
        try:
            self.assertTrue(opening.wait(5))
            model.unload("source=a").result(5)
        finally:
            release.set()

        # The open callback is posted before the worker takes this marker.
        model.inference.submit(lambda: None).result(5)
        self.assertEqual(model.held().result(5), {"models": [], "measures": []})
        settle(model)
        (failed,) = client.kinds("failed")
        self.assertIn("a was unloaded", failed.message)
        self.assertEqual(client.answered(0), [])
        self.assertEqual(first.calls, [])

        model.load("source=a").result(5)
        settle(model, until=lambda: bool(client.answered(0)))
        self.assertEqual(client.answered(0)[:2], [1, 0])
        self.assertEqual(model.held().result(5)["measures"], ["source=a"])

    def test_a_node_the_server_never_learned_is_said_unknown(self):
        model, _, measures = fixture()
        client = Received(model)
        model.demand(client.client, demand(view=(node_id([3, 3]), 10.0, 1.0))).result(5)
        settle(model)
        self.assertEqual(client.kinds("unknown"), [node_id([3, 3])])
        self.assertEqual(measures["source=a"].calls, [])


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.model, application, self.measures = fixture()
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
            refused = demand(view=(0, 10.0, 1.0))
            refused.view.rows = math.nan
            socket.send_bytes(refused.SerializeToString())
            with self.assertRaises(WebSocketDisconnect) as closed:
                socket.receive_bytes()
        self.assertEqual(closed.exception.code, 1007)

    def test_models_load_and_unload_and_measures_open_by_spec(self):
        self.assertIn("load source=c", self.client.post("/measures", json={"spec": "source=c"}).json()["message"])
        self.assertEqual(self.client.post("/models", json={"spec": "source=c"}).json()["models"][-1]["spec"], "source=c")
        wait([self.model.models["source=c"]], 5)
        self.assertEqual(self.client.post("/measures", json={"spec": "source=c|temperature=2"}).status_code, 200)
        self.model.inference.submit(lambda: None).result(5)
        self.assertIn("source=c|temperature=2", self.client.get("/held").json()["measures"])
        self.assertEqual(self.client.post("/measures", json={"spec": "source=c|depth=2"}).status_code, 400)
        self.assertEqual(self.client.post("/models", json={"spec": "source=nothing"}).status_code, 400)
        held = self.client.request("DELETE", "/models", json={"spec": "source=c"}).json()
        self.assertNotIn("source=c", [loaded["spec"] for loaded in held["models"]])
        self.assertNotIn("source=c|temperature=2", held["measures"])

    def test_search_teaches_the_server_the_ids_along_the_text(self):
        response = self.client.post(
            "/natlas.Natlas/Search",
            content=pb.SearchRequest(text="abc").SerializeToString(),
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(pb.SearchResponse.FromString(response.content).path), [0, 1, 2])
        self.assertEqual(self.measures["source=a"].calls, [])
        with self.client.websocket_connect("/session") as socket:
            socket.send_bytes(demand(path=(node_id([0, 1]), [2])).SerializeToString())
            self.updates_until_placed(socket)
        self.assertEqual(self.measures["source=a"].calls, [(99, 0, 1)])
        response = self.client.post(
            "/natlas.Natlas/Search",
            content=b"\x80",
            headers={"content-type": "application/proto"},
        )
        self.assertEqual(response.status_code, 400)


class MigrationTests(unittest.TestCase):
    def test_edge_recordings_become_weighted_runs_once(self):
        spec = importlib.util.spec_from_file_location(
            "migration", ROOT / "scripts/migrate-snapshots.py"
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        Snapshot = migration.edge_schema()
        old = Snapshot()
        recording = old.recordings.add(path=[0xFFFFFFFF])
        run = recording.children.runs.add(branches=[3, 1])
        run.ln_below.extend([0.0, math.log(0.5), math.log(0.25)])
        recording.children.count = 7
        converted = migration.convert(old.SerializeToString(), Snapshot)
        (current,) = pb.Snapshot.FromString(converted).recordings
        self.assertEqual(list(current.path), [0xFFFFFFFF])
        runs = current.children.runs
        self.assertEqual([branch for run in runs for branch in run.branches], [3, 1])
        self.assertEqual(current.children.count, 7)
        weights = [math.exp(ln) for run in runs for ln in run.ln_weights]
        self.assertAlmostEqual(weights[0], 0.5)
        self.assertAlmostEqual(weights[1], 0.25)
        self.assertEqual(runs[0].top, bytes(len(runs[0].top)))
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
