"""Run with python -m unittest discover -s tests -p '*_test.py'."""

import math
import sys
import types
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from quantile import CDF, Service, request, response, MAX_PREFIX
from natlas.v2 import natlas_pb2 as pb


class QuantileTests(unittest.TestCase):
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

    def test_service(self):
        service = Service(lambda path: [1, 1] if not path else [])
        result = pb.QuantileResponse.FromString(service.query(b""))
        self.assertEqual(len(result.delta), 256)
        child = pb.QuantileRequest(node=pb.Branch(parent=result.node, branch_id=0))
        result = pb.QuantileResponse.FromString(service.query(child.SerializeToString()))
        self.assertTrue(result.terminal)

    def test_handle_identity_does_not_depend_on_registration_order(self):
        services = [Service(lambda path: [1, 1, 1, 1]) for _ in range(2)]
        handles = []
        for service, order in zip(services, ([0, 1], [1, 0])):
            found = {}
            for label in order:
                query = pb.QuantileRequest(node=pb.Branch(parent=0, branch_id=label))
                found[label] = pb.QuantileResponse.FromString(service.query(query.SerializeToString())).node
            handles.append(found)
        self.assertEqual(handles[0], handles[1])


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


if __name__ == "__main__":
    unittest.main()
