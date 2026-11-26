from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _single_node(model: onnx.ModelProto, op_type: str, domain: str = "") -> onnx.NodeProto:
    matches = [
        node
        for node in model.graph.node
        if node.op_type == op_type and node.domain == domain
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {domain or 'ai.onnx'}::{op_type} node, found {len(matches)}")
    return matches[0]


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }


def _initializer(
    initializers: dict[str, np.ndarray], name: str
) -> np.ndarray:
    try:
        return initializers[name]
    except KeyError as error:
        raise ValueError(f"Required ONNX initializer {name!r} is missing") from error


def _scalar(initializers: dict[str, np.ndarray], name: str) -> float:
    value = _initializer(initializers, name)
    if value.size != 1:
        raise ValueError(f"Expected scalar initializer {name!r}, got shape {value.shape}")
    return float(value.reshape(()))


def _scalar_int(initializers: dict[str, np.ndarray], name: str) -> int:
    value = _initializer(initializers, name)
    if value.size != 1:
        raise ValueError(f"Expected scalar initializer {name!r}, got shape {value.shape}")
    return int(value.reshape(()))


@dataclass(frozen=True)
class QuantizedFullyConnected:
    input_scale: float
    input_zero_point: int
    weights: np.ndarray
    weight_scales: np.ndarray
    weight_zero_points: np.ndarray
    matmul_output_scale: float
    matmul_output_zero_point: int
    bias: np.ndarray
    bias_scale: float
    bias_zero_point: int
    output_scale: float
    output_zero_point: int

    @classmethod
    def from_onnx(cls, model_path: str | Path) -> "QuantizedFullyConnected":
        model = onnx.load(str(model_path))
        matmul = _single_node(model, "QLinearMatMul")
        bias_add = _single_node(model, "QLinearAdd", "com.microsoft")
        if len(matmul.input) != 8 or len(bias_add.input) != 8:
            raise ValueError("Unexpected quantized fully connected node signature")
        if not matmul.output or not bias_add.input or bias_add.input[0] != matmul.output[0]:
            raise ValueError("Expected QLinearAdd immediately after QLinearMatMul")

        initializers = _initializer_map(model)
        weights = _initializer(initializers, matmul.input[3])
        bias = _initializer(initializers, bias_add.input[3])
        if weights.ndim != 2 or bias.shape != (1, weights.shape[1]):
            raise ValueError(
                f"Unexpected fully connected parameter shapes: weights={weights.shape}, "
                f"bias={bias.shape}"
            )
        if (
            _initializer(initializers, matmul.input[4]).shape != (weights.shape[1],)
            or _initializer(initializers, matmul.input[5]).shape != (weights.shape[1],)
        ):
            raise ValueError("Expected per-output-channel weight scales and zero points")
        if weights.dtype != np.int8 or bias.dtype != np.uint8:
            raise ValueError(
                f"Unexpected parameter dtypes: weights={weights.dtype}, bias={bias.dtype}"
            )

        return cls(
            input_scale=_scalar(initializers, matmul.input[1]),
            input_zero_point=_scalar_int(initializers, matmul.input[2]),
            weights=weights.copy(),
            weight_scales=_initializer(initializers, matmul.input[4]).astype(np.float32),
            weight_zero_points=_initializer(initializers, matmul.input[5]).astype(np.int32),
            matmul_output_scale=_scalar(initializers, matmul.input[6]),
            matmul_output_zero_point=_scalar_int(initializers, matmul.input[7]),
            bias=bias.copy(),
            bias_scale=_scalar(initializers, bias_add.input[4]),
            bias_zero_point=_scalar_int(initializers, bias_add.input[5]),
            output_scale=_scalar(initializers, bias_add.input[6]),
            output_zero_point=_scalar_int(initializers, bias_add.input[7]),
        )

    @property
    def input_size(self) -> int:
        return self.weights.shape[0]

    @property
    def output_size(self) -> int:
        return self.weights.shape[1]

    def run(self, activations: np.ndarray) -> np.ndarray:
        values = np.asarray(activations)
        if values.shape == (1, self.input_size):
            values = values[0]
        if values.shape != (self.input_size,):
            raise ValueError(
                f"Expected activations with shape ({self.input_size},), got {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"Expected integer activations, got {values.dtype}")
        if np.any(values < 0) or np.any(values > 255):
            raise ValueError("Quantized uint8 activations must be in the range [0, 255]")

        centered_input = values.astype(np.int32) - np.int32(self.input_zero_point)
        centered_weights = self.weights.astype(np.int32) - self.weight_zero_points
        accumulator64 = centered_input.astype(np.int64) @ centered_weights.astype(np.int64)
        limits = np.iinfo(np.int32)
        if np.any(accumulator64 < limits.min) or np.any(accumulator64 > limits.max):
            raise OverflowError("Fully connected accumulator exceeds int32 range")
        accumulator = accumulator64.astype(np.int32)

        matmul_scale = (
            np.float32(self.input_scale)
            * self.weight_scales
            / np.float32(self.matmul_output_scale)
        )
        matmul_values = (
            np.rint(accumulator.astype(np.float32) * matmul_scale)
            + self.matmul_output_zero_point
        )
        quantized_matmul = np.clip(matmul_values, 0, 255).astype(np.uint8)

        matmul_real = (
            quantized_matmul.astype(np.float32) - np.float32(self.matmul_output_zero_point)
        ) * np.float32(self.matmul_output_scale)
        bias_real = (
            self.bias.astype(np.float32) - np.float32(self.bias_zero_point)
        ) * np.float32(self.bias_scale)
        output_values = (
            np.rint((matmul_real + bias_real) / np.float32(self.output_scale))
            + self.output_zero_point
        )
        return np.clip(output_values, 0, 255).astype(np.uint8).reshape(self.output_size)
