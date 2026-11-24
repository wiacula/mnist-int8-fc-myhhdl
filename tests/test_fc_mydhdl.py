import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from myhdl import (
    ResetSignal,
    Signal,
    Simulation,
    StopSimulation,
    always,
    delay,
    instance,
    intbv,
)

from mnist_fc.fc_mydhdl import HardwareConstants, fc_serial
from mnist_fc.fc_reference import QuantizedFullyConnected


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist-12-int8.onnx"


def _simulate_layer(
    activations: np.ndarray,
    constants: HardwareConstants,
) -> tuple[list[int], int]:
    outputs: list[int] = []
    elapsed_cycles = [0]
    clk = Signal(bool(0))
    reset = ResetSignal(0, active=1, isasync=False)
    input_data = Signal(intbv(0)[8:])
    input_valid = Signal(bool(0))
    input_ready = Signal(bool(0))
    output_data = Signal(intbv(0)[8:])
    output_valid = Signal(bool(0))
    output_ready = Signal(bool(1))

    dut = fc_serial(
        clk,
        reset,
        input_data,
        input_valid,
        input_ready,
        output_data,
        output_valid,
        output_ready,
        constants.weights,
        constants.matmul_multipliers,
        constants.add_multiplier,
        constants.add_bias_offsets,
        constants.matmul_output_zero_point,
        constants.output_zero_point,
        constants.input_size,
        constants.output_size,
    )

    @always(delay(5))
    def clock_driver():
        clk.next = not clk

    @instance
    def stimulus():
        reset.next = True
        yield clk.posedge
        yield delay(1)
        reset.next = False

        for value in activations:
            yield clk.negedge
            input_data.next = int(value)
            input_valid.next = True
            yield clk.posedge
            yield delay(1)
        input_valid.next = False

        for cycle in range(5000):
            yield clk.negedge
            if output_valid:
                outputs.append(int(output_data))
            yield clk.posedge
            yield delay(1)
            elapsed_cycles[0] = cycle + 1
            if len(outputs) == constants.output_size:
                raise StopSimulation
        raise AssertionError("Timed out waiting for all fully connected outputs")

    Simulation(dut, clock_driver, stimulus).run()
    return outputs, elapsed_cycles[0]


class MyHDLFullyConnectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layer = QuantizedFullyConnected.from_onnx(MODEL_PATH)
        cls.constants = HardwareConstants.from_layer(cls.layer)

    def test_matches_reference_for_zero_input(self) -> None:
        activations = np.zeros(self.layer.input_size, dtype=np.uint8)
        outputs, cycles = _simulate_layer(activations, self.constants)
        np.testing.assert_array_equal(outputs, self.layer.run(activations))
        self.assertEqual(cycles, self.layer.output_size * 260)

    def test_matches_reference_for_random_input(self) -> None:
        activations = np.random.default_rng(42).integers(
            0, 256, size=self.layer.input_size, dtype=np.uint8
        )
        outputs, _ = _simulate_layer(activations, self.constants)
        np.testing.assert_array_equal(outputs, self.layer.run(activations))

    def test_matches_reference_for_maximum_input(self) -> None:
        activations = np.full(self.layer.input_size, 255, dtype=np.uint8)
        outputs, _ = _simulate_layer(activations, self.constants)
        np.testing.assert_array_equal(outputs, self.layer.run(activations))

    def test_fixed_point_requantization_matches_over_full_ranges(self) -> None:
        shift_factor = 1 << 40
        half = shift_factor >> 1

        def round_even(numerators: np.ndarray) -> np.ndarray:
            quotient, remainder = np.divmod(numerators, shift_factor)
            return quotient + (
                (remainder > half)
                | ((remainder == half) & ((quotient & 1) != 0))
            )

        matmul_scales = (
            np.float32(self.layer.input_scale)
            * self.layer.weight_scales
            / np.float32(self.layer.matmul_output_scale)
        )
        for output_index in range(self.layer.output_size):
            weights = self.layer.weights[:, output_index].astype(np.int64)
            low = int(np.minimum(weights * 0, weights * 255).sum())
            high = int(np.maximum(weights * 0, weights * 255).sum())
            accumulators = np.arange(low, high + 1, dtype=np.int64)
            expected = np.clip(
                np.rint(accumulators * matmul_scales[output_index])
                + self.layer.matmul_output_zero_point,
                0,
                255,
            )
            actual = np.clip(
                round_even(
                    accumulators * self.constants.matmul_multipliers[output_index]
                )
                + self.layer.matmul_output_zero_point,
                0,
                255,
            )
            np.testing.assert_array_equal(actual, expected)

            quantized_matmul = np.arange(256, dtype=np.int64)
            bias = int(self.layer.bias.reshape(self.layer.output_size)[output_index])
            expected_real = (
                (quantized_matmul.astype(np.float32)
                 - np.float32(self.layer.matmul_output_zero_point))
                * np.float32(self.layer.matmul_output_scale)
                + (np.float32(bias) - np.float32(self.layer.bias_zero_point))
                * np.float32(self.layer.bias_scale)
            )
            expected_add = np.clip(
                np.rint(expected_real / np.float32(self.layer.output_scale))
                + self.layer.output_zero_point,
                0,
                255,
            )
            actual_add = np.clip(
                round_even(
                    (quantized_matmul - self.layer.matmul_output_zero_point)
                    * self.constants.add_multiplier
                    + self.constants.add_bias_offsets[output_index]
                )
                + self.layer.output_zero_point,
                0,
                255,
            )
            np.testing.assert_array_equal(actual_add, expected_add)

    def test_converts_to_synchronous_memory_verilog(self) -> None:
        clk = Signal(bool(0))
        reset = ResetSignal(0, active=1, isasync=False)
        input_data = Signal(intbv(0)[8:])
        input_valid = Signal(bool(0))
        input_ready = Signal(bool(0))
        output_data = Signal(intbv(0)[8:])
        output_valid = Signal(bool(0))
        output_ready = Signal(bool(0))
        dut = fc_serial(
            clk,
            reset,
            input_data,
            input_valid,
            input_ready,
            output_data,
            output_valid,
            output_ready,
            self.constants.weights,
            self.constants.matmul_multipliers,
            self.constants.add_multiplier,
            self.constants.add_bias_offsets,
            self.constants.matmul_output_zero_point,
            self.constants.output_zero_point,
            self.constants.input_size,
            self.constants.output_size,
        )
        with TemporaryDirectory() as output_dir:
            dut.convert(hdl="Verilog", path=output_dir, name="mnist_fc_serial")
            rtl = (Path(output_dir) / "mnist_fc_serial.v").read_text()

        self.assertIn("reg signed [7:0]  even_weight_memory [0:1280-1];", rtl)
        self.assertIn("reg signed [7:0]  odd_weight_memory [0:1280-1];", rtl)
        self.assertIn("always @(posedge clk) begin: read_memories", rtl)
        self.assertIn("wire signed [63:0] matmul_product;", rtl)

    def test_converter_requests_block_rom_inference_for_both_weight_banks(self) -> None:
        with TemporaryDirectory() as output_dir:
            from mnist_fc.convert_fc import main
            from unittest.mock import patch

            with patch("sys.argv", ["convert_fc.py", "--output-dir", output_dir]):
                main()
            rtl = (Path(output_dir) / "mnist_fc_serial.v").read_text()

        self.assertIn(
            '(* rom_style = "block" *)\n'
            'reg signed [7:0]  even_weight_memory [0:1280-1];',
            rtl,
        )
        self.assertIn(
            '(* rom_style = "block" *)\n'
            'reg signed [7:0]  odd_weight_memory [0:1280-1];',
            rtl,
        )
        self.assertIn(
            '$readmemh("even_weight_memory.mem", even_weight_memory);',
            rtl,
        )
        self.assertIn(
            '$readmemh("odd_weight_memory.mem", odd_weight_memory);',
            rtl,
        )


if __name__ == "__main__":
    unittest.main()
