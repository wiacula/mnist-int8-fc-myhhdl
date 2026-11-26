import argparse
import copy
import gzip
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
FLOAT_MODEL = ROOT / "models" / "mnist-12.onnx"
INT8_MODEL = ROOT / "models" / "mnist-12-int8.onnx"
FEATURE_OUTPUT = "Pooling160_Output_0_reshape0_gemm_MatMul_QuantizeLinear"
DATA_FILES = {
    "t10k-images-idx3-ubyte.gz": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/"
        "t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/"
        "t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
}

sys.path.insert(0, str(ROOT))

from mnist_fc.fc_mydhdl import HardwareConstants
from mnist_fc.fc_reference import QuantizedFullyConnected


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_md5) in DATA_FILES.items():
        path = data_dir / filename
        if path.is_file() and _md5(path) == expected_md5:
            continue

        temporary_path = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(
            url, headers={"User-Agent": "mnist-int8-fc-evaluation/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary_path.write_bytes(response.read())
            actual_md5 = _md5(temporary_path)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"Checksum mismatch for {filename}: "
                    f"expected {expected_md5}, got {actual_md5}"
                )
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _load_test_set(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    image_path = data_dir / "t10k-images-idx3-ubyte.gz"
    label_path = data_dir / "t10k-labels-idx1-ubyte.gz"
    _download_dataset(data_dir)

    with gzip.open(image_path, "rb") as stream:
        image_bytes = stream.read()
    with gzip.open(label_path, "rb") as stream:
        label_bytes = stream.read()

    image_magic, image_count, rows, columns = struct.unpack(">IIII", image_bytes[:16])
    label_magic, label_count = struct.unpack(">II", label_bytes[:8])
    if (image_magic, label_magic) != (2051, 2049):
        raise ValueError("Unexpected MNIST IDX file magic number")
    if image_count != label_count or (rows, columns) != (28, 28):
        raise ValueError("Unexpected MNIST test-set dimensions")

    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16)
    images = images.reshape(image_count, rows, columns)
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8)
    return images, labels


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def _int8_session_with_features(layer: QuantizedFullyConnected) -> ort.InferenceSession:
    model = onnx.load(str(INT8_MODEL))
    matmul = next(node for node in model.graph.node if node.op_type == "QLinearMatMul")
    bias_add = next(
        node
        for node in model.graph.node
        if node.op_type == "QLinearAdd" and node.domain == "com.microsoft"
    )
    final_quantized_output = bias_add.output[0]

    additional_outputs = (
        (FEATURE_OUTPUT, onnx.TensorProto.UINT8, [1, layer.input_size]),
        (
            final_quantized_output,
            onnx.TensorProto.UINT8,
            [1, layer.output_size],
        ),
    )
    existing_outputs = {output.name for output in model.graph.output}
    for name, element_type, shape in additional_outputs:
        if name not in existing_outputs:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, element_type, shape)
            )

    if matmul.input[0] != FEATURE_OUTPUT:
        raise ValueError("The ONNX fully connected input changed; update feature extraction")
    onnx.checker.check_model(model)
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )


def _validate_preprocessing() -> None:
    model = onnx.load(str(INT8_MODEL))
    initializers = {
        tensor.name: onnx.numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }
    quantize = next(
        node
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input[0] == model.graph.input[0].name
    )
    input_scale = float(initializers[quantize.input[1]].reshape(()))
    input_zero_point = int(initializers[quantize.input[2]].reshape(()))
    if input_scale != 1.0 or input_zero_point != 0:
        raise ValueError(
            "Expected MNIST-12-int8 input quantization scale=1 and zero point=0; "
            "review image preprocessing before evaluating"
        )


def evaluate(
    limit: int | None,
    data_dir: Path,
    myhdl_samples: int,
) -> dict[str, object]:
    if not FLOAT_MODEL.is_file() or not INT8_MODEL.is_file():
        raise FileNotFoundError("Both MNIST-12 float and int8 ONNX models are required")
    _validate_preprocessing()

    images, labels = _load_test_set(data_dir)
    sample_count = len(labels) if limit is None else min(limit, len(labels))
    if sample_count <= 0:
        raise ValueError("--limit must be positive")
    if myhdl_samples < 0:
        raise ValueError("--myhdl-samples cannot be negative")

    float_session = ort.InferenceSession(
        str(FLOAT_MODEL),
        sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )
    int8_session = _int8_session_with_features(
        QuantizedFullyConnected.from_onnx(INT8_MODEL)
    )
    float_input_name = float_session.get_inputs()[0].name
    int8_input_name = int8_session.get_inputs()[0].name
    float_input_shape = float_session.get_inputs()[0].shape
    int8_input_shape = int8_session.get_inputs()[0].shape
    if float_input_shape != [1, 1, 28, 28] or int8_input_shape != [1, 1, 28, 28]:
        raise ValueError("Expected both ONNX models to accept [1, 1, 28, 28] images")

    layer = QuantizedFullyConnected.from_onnx(INT8_MODEL)
    hardware_constants = HardwareConstants.from_layer(layer)
    matmul_scales = (
        np.float32(layer.input_scale)
        * layer.weight_scales
        / np.float32(layer.matmul_output_scale)
    )
    centered_weights = layer.weights.astype(np.int32) - layer.weight_zero_points
    float_correct = 0
    int8_correct = 0
    fc_reference_correct = 0
    float_int8_agreement = 0
    fc_reference_int8_agreement = 0
    exact_fc_vectors = 0
    exact_fc_bytes = 0
    mismatch_examples: list[int] = []
    mismatch_details: list[dict[str, object]] = []
    float32_rounding_edge_indices: list[int] = []
    real_vectors: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    real_vector_indices = set(
        np.linspace(
            0,
            sample_count - 1,
            num=min(myhdl_samples, sample_count),
            dtype=np.int64,
        ).tolist()
    )

    for index in range(sample_count):
        image = images[index].astype(np.float32)
        float_tensor = (image / np.float32(255.0)).reshape(1, 1, 28, 28)
        int8_tensor = image.reshape(1, 1, 28, 28)
        float_output = float_session.run(
            None, {float_input_name: float_tensor}
        )[0].reshape(-1)
        int8_output, features, quantized_output = int8_session.run(
            [
                "Plus214_Output_0",
                FEATURE_OUTPUT,
                "Plus214_Output_0_QuantizeLinear_0",
            ],
            {int8_input_name: int8_tensor},
        )

        label = int(labels[index])
        float_prediction = int(np.argmax(float_output))
        int8_prediction = int(np.argmax(int8_output.reshape(-1)))
        fc_output = layer.run(features.reshape(-1))
        quantized_output = quantized_output.reshape(-1)
        centered_features = (
            features.reshape(-1).astype(np.int32) - layer.input_zero_point
        )
        accumulators = centered_features.astype(np.int64) @ centered_weights.astype(
            np.int64
        )
        float32_matmul = np.rint(
            accumulators.astype(np.float32) * matmul_scales
        )
        exact_matmul = np.rint(
            accumulators.astype(np.float64) * matmul_scales.astype(np.float64)
        )
        rounding_edge = bool(np.any(float32_matmul != exact_matmul))
        if rounding_edge:
            float32_rounding_edge_indices.append(index)

        float_correct += float_prediction == label
        int8_correct += int8_prediction == label
        fc_reference_correct += int(np.argmax(fc_output)) == label
        float_int8_agreement += float_prediction == int8_prediction
        fc_reference_int8_agreement += int(np.argmax(fc_output)) == int8_prediction
        exact = np.array_equal(fc_output, quantized_output)
        exact_fc_vectors += exact
        exact_fc_bytes += int(np.count_nonzero(fc_output == quantized_output))
        if not exact:
            if len(mismatch_examples) < 10:
                mismatch_examples.append(index)
            mismatch_details.append(
                {
                    "sample_index": index,
                    "label": label,
                    "mismatched_output_indices": np.flatnonzero(
                        fc_output != quantized_output
                    ).tolist(),
                    "fc_reference_output": fc_output.tolist(),
                    "int8_model_output": quantized_output.tolist(),
                }
            )
        if index in real_vector_indices or (
            myhdl_samples > 0 and (not exact or rounding_edge)
        ):
            real_vectors.append(
                (
                    index,
                    features.reshape(-1),
                    quantized_output,
                    fc_output,
                    label,
                )
            )

        if (index + 1) % 1000 == 0:
            print(f"Evaluated {index + 1}/{sample_count} MNIST samples")

    myhdl_results: dict[str, dict[str, int]] = {}
    if real_vectors:
        from tests.test_fc_mydhdl import _simulate_layer

        for name, parallel in (("P=1", False), ("P=2", True)):
            exact_onnx_count = 0
            exact_reference_count = 0
            correct_count = 0
            for _, features, expected_onnx, expected_reference, label in real_vectors:
                actual, _ = _simulate_layer(
                    features, hardware_constants, parallel=parallel
                )
                exact_onnx_count += np.array_equal(actual, expected_onnx)
                exact_reference_count += np.array_equal(actual, expected_reference)
                correct_count += int(np.argmax(actual) == label)
            myhdl_results[name] = {
                "real_mnist_vectors": len(real_vectors),
                "byte_exact_int8_model_vectors": exact_onnx_count,
                "byte_exact_fc_reference_vectors": exact_reference_count,
                "correct_predictions": correct_count,
                "top1_accuracy_percent": round(
                    100.0 * correct_count / len(real_vectors), 4
                ),
            }

    result: dict[str, object] = {
        "dataset": "MNIST official t10k test split",
        "samples": sample_count,
        "dataset_file_md5": {
            filename: _md5(data_dir / filename) for filename in DATA_FILES
        },
        "models_sha256": {
            FLOAT_MODEL.name: _sha256(FLOAT_MODEL),
            INT8_MODEL.name: _sha256(INT8_MODEL),
        },
        "input_preprocessing": {
            "float_model": "float32 pixels in [0, 1], divided by 255",
            "int8_model": "float32 pixels in [0, 255], scale=1, zero_point=0",
        },
        "float_top1_accuracy_percent": round(100.0 * float_correct / sample_count, 4),
        "int8_top1_accuracy_percent": round(100.0 * int8_correct / sample_count, 4),
        "fc_reference_top1_accuracy_percent": round(
            100.0 * fc_reference_correct / sample_count, 4
        ),
        "float_int8_prediction_agreement_percent": round(
            100.0 * float_int8_agreement / sample_count, 4
        ),
        "fc_reference_int8_prediction_agreement_percent": round(
            100.0 * fc_reference_int8_agreement / sample_count, 4
        ),
        "fc_reference_exact_output_vectors": exact_fc_vectors,
        "fc_reference_exact_output_bytes": exact_fc_bytes,
        "fc_reference_output_mismatch_examples": mismatch_examples,
        "fc_reference_output_mismatches": mismatch_details,
        "myhdl_simulation_on_real_mnist_vectors": myhdl_results,
        "float32_rounding_edge_indices": float32_rounding_edge_indices,
        "myhdl_sample_indices": [vector[0] for vector in real_vectors],
        "onnxruntime_version": ort.__version__,
    }
    return result


def main() -> None:
    default_data_dir = Path(
        os.environ.get("LOCALAPPDATA", str(Path.home()))
    ) / "mnist-int8-fc" / "mnist"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the MNIST-12 float/int8 models and the 256-to-10 "
            "fully connected layer on the official MNIST test split."
        )
    )
    parser.add_argument("--limit", type=int, help="evaluate only the first N test images")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="cache directory for the official MNIST test files",
    )
    parser.add_argument(
        "--myhdl-samples",
        type=int,
        default=10,
        help="also cycle-simulate this many real images on P=1 and P=2 (default: 10)",
    )
    parser.add_argument("--json-out", type=Path, help="write the summary to a JSON file")
    args = parser.parse_args()

    result = evaluate(args.limit, args.data_dir, args.myhdl_samples)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
