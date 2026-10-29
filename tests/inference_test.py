"""Deterministic worker/lifecycle tests, with no model or device dependencies."""

import asyncio
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from inference import Busy, Closed, Job, LocalBackend, Worker


class Backend:
    def __init__(self):
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.error = None

    async def evaluate(self, jobs):
        self.calls.append(jobs)
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        # Deliberately return results in reverse order, as a remote backend may.
        return {job: (float(job.measure), float(sum(job.path))) for job in reversed(jobs)}

    async def aclose(self):
        self.closed = True


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = Backend()
        self.worker = Worker(self.backend, capacity=3, batch_size=2)
        self.worker.start()
        self.addAsyncCleanup(self.worker.aclose)
        # Always release a failed assertion's fake inference before cleanup.
        self.addCleanup(self.backend.release.set)

    async def entered(self):
        await asyncio.wait_for(self.backend.entered.wait(), 2)

    async def test_duplicates_share_inference_and_one_cancel_does_not_cancel_the_other(self):
        job = Job(0, (1, 2))
        first = asyncio.create_task(self.worker.infer(job))
        await self.entered()
        second = asyncio.create_task(self.worker.infer(job))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertFalse(second.done())
        self.backend.release.set()
        self.assertEqual(await second, (0.0, 3.0))
        self.assertEqual(self.backend.calls, [(job,)])

    async def test_admission_is_bounded_and_duplicates_still_join_at_capacity(self):
        jobs = [Job(0, (i,)) for i in range(4)]
        tasks = [asyncio.create_task(self.worker.infer(job)) for job in jobs[:3]]
        await self.entered()
        with self.assertRaises(Busy):
            await self.worker.infer(jobs[3])
        duplicate = asyncio.create_task(self.worker.infer(jobs[0]))
        await asyncio.sleep(0)
        self.backend.release.set()
        self.assertEqual(await duplicate, (0.0, 0.0))
        await asyncio.gather(*tasks)

    async def test_cancelled_queued_job_frees_capacity_and_never_reaches_backend(self):
        active = asyncio.create_task(self.worker.infer(Job(0, (0,))))
        await self.entered()
        abandoned = asyncio.create_task(self.worker.infer(Job(0, (1,))))
        await asyncio.sleep(0)
        abandoned.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await abandoned
        later = [Job(0, (2,)), Job(0, (3,))]
        tasks = [asyncio.create_task(self.worker.infer(job)) for job in later]
        await asyncio.sleep(0)
        self.assertEqual(len(self.worker.pending), 3)
        self.backend.release.set()
        await active
        await asyncio.gather(*tasks)
        self.assertNotIn(Job(0, (1,)), [job for batch in self.backend.calls for job in batch])

    async def test_batches_keep_measure_and_path_identity(self):
        jobs = [Job(0, (1,)), Job(1, (1,)), Job(1, (2,))]
        tasks = [asyncio.create_task(self.worker.infer(job)) for job in jobs]
        await self.entered()
        self.backend.release.set()
        self.assertEqual(await asyncio.gather(*tasks), [(0.0, 1.0), (1.0, 1.0), (1.0, 2.0)])
        self.assertEqual(self.backend.calls, [tuple(jobs[:2]), tuple(jobs[2:])])

    async def test_backend_failure_settles_waiters_and_next_job_can_succeed(self):
        job = Job(0, ())
        failed = asyncio.create_task(self.worker.infer(job))
        await self.entered()
        self.backend.error = ValueError("bad batch")
        self.backend.release.set()
        with self.assertRaisesRegex(ValueError, "bad batch"):
            await failed
        self.backend.error = None
        self.assertEqual(await self.worker.infer(job), (0.0, 0.0))

    async def test_incomplete_backend_batch_is_rejected_and_worker_recovers(self):
        original = self.backend.evaluate

        async def incomplete(jobs):
            return {}

        self.backend.evaluate = incomplete
        with self.assertRaisesRegex(RuntimeError, "different set of jobs"):
            await self.worker.infer(Job(0, ()))
        self.backend.evaluate = original
        self.backend.release.set()
        self.assertEqual(await self.worker.infer(Job(0, ())), (0.0, 0.0))

    async def test_one_failed_job_does_not_fail_other_jobs_in_the_batch(self):
        jobs = [Job(0, (0,)), Job(0, (1,))]

        async def mixed(batch):
            return {job: ValueError("bad node") if job == jobs[0] else (1.0,) for job in batch}

        self.backend.evaluate = mixed
        results = await asyncio.gather(*(self.worker.infer(job) for job in jobs), return_exceptions=True)
        self.assertIsInstance(results[0], ValueError)
        self.assertEqual(results[1], (1.0,))

    async def test_shutdown_rejects_queued_work_and_finishes_running_work(self):
        active = asyncio.create_task(self.worker.infer(Job(0, (0,))))
        await self.entered()
        queued = asyncio.create_task(self.worker.infer(Job(0, (1,))))
        await asyncio.sleep(0)
        closing = asyncio.create_task(self.worker.aclose())
        with self.assertRaises(Closed):
            await queued
        with self.assertRaises(Closed):
            await self.worker.infer(Job(0, (2,)))
        self.assertFalse(closing.done())
        self.assertFalse(self.backend.closed)
        self.backend.release.set()
        self.assertEqual(await active, (0.0, 0.0))
        await closing
        self.assertTrue(self.backend.closed)

    async def test_cancelling_shutdown_waiter_does_not_abandon_backend(self):
        active = asyncio.create_task(self.worker.infer(Job(0, (0,))))
        await self.entered()
        closing = asyncio.create_task(self.worker.aclose())
        await asyncio.sleep(0)
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.backend.release.set()
        await active
        await self.worker.aclose()
        self.assertTrue(self.backend.closed)

    async def test_invalid_configuration_and_submission_outside_lifespan(self):
        with self.assertRaises(ValueError):
            Worker(self.backend, capacity=0)
        with self.assertRaises(ValueError):
            Worker(self.backend, batch_size=0)
        worker = Worker(Backend())
        with self.assertRaises(Closed):
            await worker.infer(Job(0, ()))
        await worker.aclose()
        with self.assertRaises(Closed):
            worker.start()


class LocalBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_calls_use_one_dedicated_thread_and_copy_outputs(self):
        loop_thread = threading.get_ident()
        threads = []
        logits = [0.0]

        def evaluate(job):
            threads.append(threading.get_ident())
            if job.path == (2,):
                raise ValueError("bad node")
            logits[0] = float(job.measure)
            return logits

        backend = LocalBackend(evaluate)
        self.addAsyncCleanup(backend.aclose)
        jobs = (Job(1, (0,)), Job(2, (1,)), Job(3, (2,)))
        results = await backend.evaluate(jobs)
        self.assertEqual(results[jobs[0]], (1.0,))
        self.assertEqual(results[jobs[1]], (2.0,))
        self.assertIsInstance(results[jobs[2]], ValueError)
        await backend.evaluate((Job(0, (0,)),))
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], loop_thread)
