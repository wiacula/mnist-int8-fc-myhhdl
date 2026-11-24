import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from mnist_fc.fc_reference import QuantizedFullyConnected


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist-12-int8.onnx"


def _make_onnx_reference_session() -> tuple[ort.InferenceSession, str, str]:
    model = onnx.load(str(MODEL_PATH))
    matmul = next(node for node in model.graph.node if node.op_type == "QLinearMatMul")
    bias_add = next(
        node
        for node in model.graph.node
        if node.op_type == "QLinearAdd" and node.domain == "com.microsoft"
    )
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    required_initializers = {
        name
        for node in (matmul, bias_add)
        for name in node.input[1:]
    }
    graph = helper.make_graph(
        nodes=[matmul, bias_add],
        name="mnist_fc_reference",
        inputs=[
            helper.make_tensor_value_info(matmul.input[0], TensorProto.UINT8, [1, 256])
        ],
        outputs=[
            helper.make_tensor_value_info(bias_add.output[0], TensorProto.UINT8, [1, 10])
        ],
        initializer=[initializers[name] for name in sorted(required_initializers)],
    )
    reference_model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 12),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=model.ir_version,
    )
    onnx.checker.check_model(reference_model)
    session = ort.InferenceSession(
        reference_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session, matmul.input[0], bias_add.output[0]


class QuantizedFullyConnectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layer = QuantizedFullyConnected.from_onnx(MODEL_PATH)
        cls.session, cls.input_name, cls.output_name = _make_onnx_reference_session()

    def test_extracts_expected_layer_dimensions(self) -> None:
        self.assertEqual(self.layer.input_size, 256)
        self.assertEqual(self.layer.output_size, 10)

    def test_matches_onnx_runtime_for_boundary_and_random_inputs(self) -> None:
        rng = np.random.default_rng(42)
        vectors = [
            np.zeros(256, dtype=np.uint8),
            np.full(256, 255, dtype=np.uint8),
            *(
                rng.integers(0, 256, size=(256,), dtype=np.uint8)
                for _ in range(32)
            ),
        ]
        for values in vectors:
            with self.subTest(first_value=int(values[0])):
                expected = self.session.run(
                    [self.output_name],
                    {self.input_name: values.reshape(1, 256)},
                )[0].reshape(10)
                np.testing.assert_array_equal(self.layer.run(values), expected)

    def test_rejects_wrong_input_shape(self) -> None:
        with self.assertRaises(ValueError):
            self.layer.run(np.zeros(255, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
