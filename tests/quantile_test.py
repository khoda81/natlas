"""Run with python -m unittest discover -s tests -p '*_test.py'."""

import asyncio
import io
import math
import struct
import sys
import threading
import types
import unittest
from collections import Counter
from contextlib import redirect_stderr
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from quantile import CDF, Service, request, response, MAX_PREFIX
from natlas.v2 import natlas_pb2 as pb


class QuantileTests(unittest.IsolatedAsyncioTestCase):
    def test_uniform(self):
        first, deltas, terminal = CDF([1] * 256).page(b"")
        self.assertEqual((first, deltas, terminal), (0, [1] * 256, False))

    def test_zeros(self):
        cdf = CDF([0, 1, 0, 1, 0])
        self.assertEqual(cdf.at(0, 1), 1)
        self.assertEqual(cdf.at(1, 2), 3)
        self.assertEqual(cdf.at(1, 1), 5)

    def test_deep(self):
        cdf = CDF([Fraction(1, 2), Fraction(1, 2**200), Fraction(1, 2) - Fraction(1, 2**200)])
        first, deltas, _ = cdf.page(b"\x80" + b"\0" * 23)
        self.assertEqual(first, 1)
        self.assertEqual(deltas[0], 1)

    def test_underflowing_log_weights(self):
        cdf = CDF.from_log_weights([0, -2000, -math.inf])
        self.assertGreater(cdf.ends[1], cdf.ends[0])
        self.assertEqual(cdf.ends[2], cdf.ends[1])
        self.assertAlmostEqual(math.log(cdf.ends[1] - cdf.ends[0]) - math.log(cdf.total), -2000, places=10)

    def test_parent_end(self):
        cdf = CDF([1, 2, 3, 7])
        first, deltas, _ = cdf.page(b"")
        ids = [first]
        for delta in deltas:
            ids.append(ids[-1] + delta)
        for j in range(256):
            first, deltas, _ = cdf.page(bytes([j]))
            self.assertEqual(first, ids[j])
            self.assertEqual(first + sum(deltas), ids[j + 1])

    def test_terminal(self):
        self.assertEqual(CDF([]).page(b""), (0, [], True))

    def test_invalid(self):
        for weights in ([0, 0], [-1, 2], [float("nan")]):
            with self.assertRaises(ValueError):
                CDF(weights)
        with self.assertRaises(ValueError):
            request(b"\x12\xff")
        with self.assertRaises(OverflowError):
            request(pb.QuantileRequest(prefix=bytes(MAX_PREFIX + 1)).SerializeToString())
        with self.assertRaises(OverflowError):
            CDF.from_log_weights([0, -1e9])

    def test_generated_wire_format(self):
        data = response(2**63, 17, [0] * 255 + [2**32])
        result = pb.QuantileResponse.FromString(data)
        self.assertEqual(result.node, 2**63)
        self.assertEqual(result.delta[-1], 2**32)
        self.assertEqual(result.first_branch_id, 17)

    async def test_service(self):
        async def distribution(path):
            return [1, 1] if not path else []

        service = Service(distribution)
        self.addAsyncCleanup(service.aclose)
        result = pb.QuantileResponse.FromString(await service.query(b""))
        self.assertEqual(len(result.delta), 256)
        child = pb.QuantileRequest(node=pb.Branch(parent=result.node, branch_id=0))
        result = pb.QuantileResponse.FromString(await service.query(child.SerializeToString()))
        self.assertTrue(result.terminal)

    async def test_handle_identity_does_not_depend_on_registration_order(self):
        async def distribution(path):
            return [1, 1, 1, 1]

        services = [Service(distribution) for _ in range(2)]
        handles = []
        for service, order in zip(services, ([0, 1], [1, 0])):
            self.addAsyncCleanup(service.aclose)
            found = {}
            for label in order:
                query = pb.QuantileRequest(node=pb.Branch(parent=0, branch_id=label))
                found[label] = pb.QuantileResponse.FromString(await service.query(query.SerializeToString())).node
            handles.append(found)
        self.assertEqual(handles[0], handles[1])

    async def test_default_cache_keeps_a_working_set_larger_than_eight_nodes(self):
        calls = Counter()

        async def distribution(path):
            calls[path] += 1
            return [1.0] * 16

        service = Service(distribution)
        self.addAsyncCleanup(service.aclose)
        for _ in range(3):
            for label in range(16):
                message = pb.QuantileRequest(node=pb.Branch(parent=0, branch_id=label))
                await service.query(message.SerializeToString())
        self.assertEqual(calls, Counter({(): 1, **{(label,): 1 for label in range(16)}}))

    async def test_cache_evicts_by_storage_and_preserves_recently_used_cdfs(self):
        calls = Counter()
        size = CDF([1.0] * 4).memory_bytes

        async def distribution(path):
            calls[path] += 1
            return [1.0] * 4

        service = Service(distribution, cache_bytes=2 * size)
        self.addAsyncCleanup(service.aclose)
        for label in [0, 1, 0, 2]:
            await service.cdf((label,))
        self.assertEqual(set(service.cache), {(0,), (2,)})
        self.assertLessEqual(service.cached_bytes, service.cache_bytes)
        await service.cdf((0,))
        self.assertEqual(calls[(0,)], 1)
        await service.cdf((1,))
        self.assertEqual(calls[(1,)], 2)

    async def test_oversized_cdf_is_reused_for_later_pages(self):
        calls = 0

        async def distribution(path):
            nonlocal calls
            calls += 1
            return [1.0] * 4

        service = Service(distribution, cache_bytes=1)
        self.addAsyncCleanup(service.aclose)
        await service.query(b"")
        await service.query(pb.QuantileRequest(prefix=b"\x7f").SerializeToString())
        self.assertEqual(calls, 1)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real app/routes with a deterministic inference fixture.
        # Importing the heavyweight torch backends is unnecessary for this test.
        sys.modules.setdefault("lm", types.SimpleNamespace(TooLong=type("TooLong", (Exception,), {})))
        import serve
        from starlette.testclient import TestClient

        measure = types.SimpleNamespace(
            log_probs=lambda path: types.SimpleNamespace(tolist=lambda: [-1.0, -2.0, -800.0])
        )
        model = serve.Model((), lambda path: len(path) > 1, 3, [measure])
        cls.client = TestClient(serve.app(model, [], {}))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def post(self, data):
        return self.client.post(
            "/0/natlas.v2.Natlas/Quantile", content=data, headers={"content-type": "application/proto"}
        )

    def test_unary_endpoint(self):
        result = self.post(b"")
        self.assertEqual(result.status_code, 200)
        message = pb.QuantileResponse.FromString(result.content)
        self.assertEqual(len(message.delta), 256)
        self.assertEqual(message.first_branch_id + sum(message.delta), 3)

    def test_error_statuses(self):
        self.assertEqual(self.post(b"\x12\xff").status_code, 400)
        missing = pb.QuantileRequest(node=pb.Branch(parent=123456, branch_id=0))
        self.assertEqual(self.post(missing.SerializeToString()).json()["code"], "not_found")
        huge = pb.QuantileRequest(prefix=bytes(MAX_PREFIX + 1))
        self.assertEqual(self.post(huge.SerializeToString()).status_code, 429)

    def test_legacy_endpoint_remains_available(self):
        # horizon 0 resolves the handle without invoking the legacy sort API.
        result = self.client.post(
            "/0/natlas.v1.Natlas/Expand", content=b"\0\0\0\0\0", headers={"content-type": "application/connect+proto"}
        )
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.content)


class ServerConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        sys.modules.setdefault("lm", types.SimpleNamespace(TooLong=type("TooLong", (Exception,), {})))
        try:
            import httpx2 as httpx
        except ImportError:
            import httpx
        import serve

        self.httpx = httpx
        loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.calls = Counter()
        self.threads = set()
        self.fail = set()
        self.block = {(0, (1,))}

        def measure(index):
            def log_probs(path):
                key = index, path
                self.calls[key] += 1
                self.threads.add(threading.get_ident())
                if key in self.block:
                    loop.call_soon_threadsafe(self.entered.set)
                    if not self.release.wait(5):
                        raise TimeoutError("test did not release inference")
                if key in self.fail:
                    raise RuntimeError("fixture inference failed")
                return types.SimpleNamespace(tolist=lambda: [-1.0, -2.0, -800.0])

            return types.SimpleNamespace(log_probs=log_probs)

        model = serve.Model((), lambda path: len(path) > 1, 3, [measure(0), measure(1)])
        self.application = serve.app(model, [], {}, capacity=2)
        self.lifespan = self.application.router.lifespan_context(self.application)
        await self.lifespan.__aenter__()
        self.addAsyncCleanup(self.lifespan.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.application), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)
        self.addCleanup(self.release.set)
        root = await self.post()
        self.root = pb.QuantileResponse.FromString(root.content).node

    async def post(self, label=None, prefix=b"", *, measure=0, parent=None):
        node = None if label is None else pb.Branch(parent=self.root if parent is None else parent, branch_id=label)
        message = pb.QuantileRequest(node=node, prefix=prefix)
        return await self.client.post(
            f"/{measure}/natlas.v2.Natlas/Quantile",
            content=message.SerializeToString(),
            headers={"content-type": "application/proto"},
        )

    async def legacy(self, label=None, *, horizon=10):
        from natlas.v1 import natlas_pb2 as v1

        node = None if label is None else v1.Child(parent=0, label=label)
        body = v1.ExpandRequest(node=node, horizon=horizon).SerializeToString()
        return await self.client.post(
            "/0/natlas.v1.Natlas/Expand",
            content=b"\0" + len(body).to_bytes(4, "big") + body,
            headers={"content-type": "application/connect+proto"},
        )

    async def cold(self):
        task = asyncio.create_task(self.post(1))
        await asyncio.wait_for(self.entered.wait(), 2)
        return task

    async def test_cached_refinement_and_manifest_return_during_cold_inference(self):
        cold = await self.cold()
        cached = await asyncio.wait_for(self.post(prefix=b"\xff"), 2)
        self.assertEqual(cached.status_code, 200)
        manifest = await asyncio.wait_for(self.client.get("/manifest.json"), 2)
        self.assertEqual(manifest.status_code, 200)
        self.assertFalse(cold.done())
        self.assertEqual(self.calls[(0, ())], 1)
        self.release.set()
        self.assertEqual((await cold).status_code, 200)

    async def test_duplicate_pages_share_inference_and_survive_one_http_cancellation(self):
        first = await self.cold()
        second = asyncio.create_task(self.post(1, b"\xff"))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.release.set()
        result = await second
        self.assertEqual(result.status_code, 200)
        message = pb.QuantileResponse.FromString(result.content)
        expected = CDF.from_log_weights([-1, -2, -800]).page(b"\xff")
        self.assertEqual((message.first_branch_id, list(message.delta), message.terminal), expected)
        self.assertEqual((await self.post(1, b"\x00")).status_code, 200)
        self.assertEqual(self.calls[(0, (1,))], 1)

    async def test_queue_overload_returns_429_while_cache_hits_still_work(self):
        cold = await self.cold()
        queued = asyncio.create_task(self.post(2))
        # Wait for admission rather than relying on a wall-clock scheduling delay.
        for _ in range(100):
            if len(self.application.state.inference.pending) == 2:
                break
            await asyncio.sleep(0)
        self.assertEqual(len(self.application.state.inference.pending), 2)
        overloaded = await self.post(0)
        self.assertEqual(overloaded.status_code, 429)
        self.assertEqual(overloaded.json()["code"], "resource_exhausted")
        legacy = await self.legacy(0)
        self.assertIn(b'"code": "resource_exhausted"', legacy.content)
        self.assertEqual((await self.post(prefix=b"\x00")).status_code, 200)
        self.assertFalse(cold.done())
        self.release.set()
        self.assertEqual([reply.status_code for reply in await asyncio.gather(cold, queued)], [200, 200])

    async def test_legacy_cache_bypasses_busy_inference_and_protocols_share_the_worker(self):
        self.assertEqual((await self.legacy()).status_code, 200)
        cold = await self.cold()
        cached = await asyncio.wait_for(self.legacy(), 2)
        self.assertEqual(cached.status_code, 200)
        queued = asyncio.create_task(self.legacy(0))
        await asyncio.sleep(0)
        self.assertFalse(cold.done())
        self.assertFalse(queued.done())
        self.assertEqual(self.calls[(0, (0,))], 0)
        self.release.set()
        self.assertEqual((await cold).status_code, 200)
        self.assertEqual((await queued).status_code, 200)
        self.assertEqual(len(self.threads), 1)
        self.assertNotIn(threading.get_ident(), self.threads)

    async def test_different_measures_have_independent_job_keys(self):
        replies = await asyncio.gather(self.post(0), self.post(0, measure=1))
        self.assertEqual([reply.status_code for reply in replies], [200, 200])
        self.assertEqual(self.calls[(0, (0,))], 1)
        self.assertEqual(self.calls[(1, (0,))], 1)
        self.assertEqual(len(self.threads), 1)

    async def test_failed_inference_is_retryable_and_does_not_poison_the_cdf_cache(self):
        self.fail.add((0, (0,)))
        with redirect_stderr(io.StringIO()):
            failed = await self.post(0)
        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json()["code"], "internal")
        self.fail.clear()
        self.assertEqual((await self.post(0)).status_code, 200)
        self.assertEqual(self.calls[(0, (0,))], 2)

    async def test_terminal_node_returns_without_submitting_inference(self):
        child = await self.post(0)
        handle = pb.QuantileResponse.FromString(child.content).node
        before = self.calls.copy()
        result = await self.post(0, parent=handle)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(pb.QuantileResponse.FromString(result.content).terminal)
        self.assertEqual(self.calls, before)

    async def test_legacy_wire_weights_are_preserved(self):
        from natlas.v1 import natlas_pb2 as v1
        from natlas_codec import encode

        result = await self.legacy(horizon=16)
        offset, edges, handles = 0, [], []
        while offset < len(result.content):
            flags, size = struct.unpack_from(">BI", result.content, offset)
            offset += 5
            body = result.content[offset : offset + size]
            offset += size
            if flags == 0:
                message = v1.ExpandResponse.FromString(body)
                if message.HasField("edge"):
                    edges.append(message.edge)
                else:
                    handles.append(message.node)
        self.assertEqual(handles, [0])
        self.assertEqual(edges, encode([0, 1], [-1, -2], 16))

    async def test_async_backend_can_replace_the_local_model(self):
        import serve

        class RemoteFixture:
            def __init__(self):
                self.jobs = []
                self.closed = False

            async def evaluate(self, jobs):
                self.jobs.extend(jobs)
                return {job: (-1.0, -2.0, -800.0) for job in jobs}

            async def aclose(self):
                self.closed = True

        from inference import Job

        backend = RemoteFixture()
        # This measure has no local inference method at all.
        model = serve.Model((), lambda path: False, 3, [object()])
        application = serve.app(model, [], {}, backend=backend, batch_size=4)
        async with application.router.lifespan_context(application):
            async with self.httpx.AsyncClient(
                transport=self.httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                root = await client.post(
                    "/0/natlas.v2.Natlas/Quantile", content=b"", headers={"content-type": "application/proto"}
                )
                self.assertEqual(root.status_code, 200)
                handle = pb.QuantileResponse.FromString(root.content).node
                child = pb.QuantileRequest(node=pb.Branch(parent=handle, branch_id=1))
                reply = await client.post(
                    "/0/natlas.v2.Natlas/Quantile",
                    content=child.SerializeToString(),
                    headers={"content-type": "application/proto"},
                )
                self.assertEqual(reply.status_code, 200)
        self.assertEqual(backend.jobs, [Job(0, ()), Job(0, (1,))])
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
