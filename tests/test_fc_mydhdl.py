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

from mnist_fc.fc_mydhdl import (
    FIXED_SHIFT,
    HardwareConstants,
    fc_parallel2,
    fc_serial,
    float32_matmul_requantize,
)
from mnist_fc.fc_reference import QuantizedFullyConnected


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist-12-int8.onnx"


def _simulate_layer(
    activations: np.ndarray,
    constants: HardwareConstants,
    parallel: bool = False,
    stall_outputs: bool = False,
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
    output_ready = Signal(bool(not stall_outputs))

    dut_factory = fc_parallel2 if parallel else fc_serial
    dut = dut_factory(
        clk,
        reset,
        input_data,
        input_valid,
        input_ready,
        output_data,
        output_valid,
        output_ready,
        constants.weights,
        constants.matmul_scale_mantissas,
        constants.matmul_scale_exponents,
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
            output_ready.next = not stall_outputs or cycle % 4 != 0
            yield delay(1)
            if output_valid and output_ready:
                outputs.append(int(output_data))
            yield clk.posedge
            yield delay(1)
            elapsed_cycles[0] = cycle + 1
            if len(outputs) == constants.output_size:
                raise StopSimulation
        raise AssertionError("Timed out waiting for all fully connected outputs")

    Simulation(dut, clock_driver, stimulus).run()
    return outputs, elapsed_cycles[0]


def _simulate_matmul_requantizer(
    cases: list[tuple[int, int]],
    constants: HardwareConstants,
    zero_point: int,
) -> list[int]:
    outputs: list[int] = []
    accumulator = Signal(intbv(0, min=-(1 << 31), max=(1 << 31)))
    scale_mantissa = Signal(intbv(0)[24:])
    scale_exponent = Signal(intbv(0, min=-64, max=1))
    output = Signal(intbv(0)[8:])
    dut = float32_matmul_requantize(
        accumulator,
        scale_mantissa,
        scale_exponent,
        zero_point,
        output,
    )

    @instance
    def stimulus():
        for value, channel in cases:
            accumulator.next = value
            scale_mantissa.next = constants.matmul_scale_mantissas[channel]
            scale_exponent.next = constants.matmul_scale_exponents[channel]
            yield delay(1)
            outputs.append(int(output))
        raise StopSimulation

    Simulation(dut, stimulus).run()
    return outputs


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

    def test_parallel2_matches_reference_with_reduced_compute_cycles(self) -> None:
        vectors = [
            np.zeros(self.layer.input_size, dtype=np.uint8),
            np.full(self.layer.input_size, 255, dtype=np.uint8),
            np.random.default_rng(2026).integers(
                0, 256, size=self.layer.input_size, dtype=np.uint8
            ),
        ]
        for index, activations in enumerate(vectors):
            with self.subTest(vector=index):
                outputs, cycles = _simulate_layer(
                    activations,
                    self.constants,
                    parallel=True,
                )
                np.testing.assert_array_equal(outputs, self.layer.run(activations))
                if index == 0:
                    self.assertEqual(
                        cycles,
                        (self.layer.output_size // 2) * 261,
                    )

    def test_parallel2_holds_output_during_backpressure(self) -> None:
        activations = np.random.default_rng(42).integers(
            0, 256, size=self.layer.input_size, dtype=np.uint8
        )
        outputs, _ = _simulate_layer(
            activations,
            self.constants,
            parallel=True,
            stall_outputs=True,
        )
        np.testing.assert_array_equal(outputs, self.layer.run(activations))

    def test_float32_matmul_rounding_matches_onnx_near_half_cases(self) -> None:
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
        cases: list[tuple[int, int]] = []
        for output_index in range(self.layer.output_size):
            weights = self.layer.weights[:, output_index].astype(np.int64)
            low = int(np.minimum(weights * 0, weights * 255).sum())
            high = int(np.maximum(weights * 0, weights * 255).sum())
            accumulators = np.arange(low, high + 1, dtype=np.int64)
            multiplier = int(
                np.rint(float(matmul_scales[output_index]) * shift_factor)
            )
            old_product = accumulators * multiplier
            old_matmul = round_even(old_product)
            float32_matmul = np.rint(
                accumulators.astype(np.float32) * matmul_scales[output_index]
            )
            differing_accumulators = accumulators[old_matmul != float32_matmul]
            cases.extend(
                (int(value), output_index) for value in differing_accumulators
            )
            cases.extend(
                (value, output_index)
                for value in (low, -1, 0, 1, high)
                if low <= value <= high
            )

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

        self.assertIn((-37225, 6), cases)
        actual = _simulate_matmul_requantizer(
            cases,
            self.constants,
            self.layer.matmul_output_zero_point,
        )
        expected = [
            int(
                np.clip(
                    np.rint(np.float32(value) * matmul_scales[channel])
                    + self.layer.matmul_output_zero_point,
                    0,
                    255,
                )
            )
            for value, channel in cases
        ]
        np.testing.assert_array_equal(actual, expected)
        counterexample_index = cases.index((-37225, 6))
        self.assertEqual(actual[counterexample_index], 76)

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
            self.constants.matmul_scale_mantissas,
            self.constants.matmul_scale_exponents,
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
        self.assertIn("matmul_scale_mantissa_memory", rtl)

    def test_converter_requests_block_rom_inference_for_both_weight_banks(self) -> None:
        with TemporaryDirectory() as output_dir:
            from mnist_fc.convert_fc import main
            from unittest.mock import patch

            with patch("sys.argv", ["convert_fc.py", "--output-dir", output_dir]):
                main()
            rtl = (Path(output_dir) / "mnist_fc_serial.v").read_text()

        self.assertIn(
            '(* rom_style = "block", ram_style = "block" *)\n'
            'reg signed [7:0]  even_weight_memory [0:1280-1];',
            rtl,
        )
        self.assertIn(
            '(* rom_style = "block", ram_style = "block" *)\n'
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

    def test_parallel2_converts_with_packed_weight_memory(self) -> None:
        clk = Signal(bool(0))
        reset = ResetSignal(0, active=1, isasync=False)
        input_data = Signal(intbv(0)[8:])
        input_valid = Signal(bool(0))
        input_ready = Signal(bool(0))
        output_data = Signal(intbv(0)[8:])
        output_valid = Signal(bool(0))
        output_ready = Signal(bool(0))
        dut = fc_parallel2(
            clk,
            reset,
            input_data,
            input_valid,
            input_ready,
            output_data,
            output_valid,
            output_ready,
            self.constants.weights,
            self.constants.matmul_scale_mantissas,
            self.constants.matmul_scale_exponents,
            self.constants.add_multiplier,
            self.constants.add_bias_offsets,
            self.constants.matmul_output_zero_point,
            self.constants.output_zero_point,
            self.constants.input_size,
            self.constants.output_size,
        )
        with TemporaryDirectory() as output_dir:
            dut.convert(hdl="Verilog", path=output_dir, name="mnist_fc_parallel2")
            rtl = (Path(output_dir) / "mnist_fc_parallel2.v").read_text()

        self.assertIn("reg [15:0]  pair_weight_memory [0:1280-1];", rtl)
        self.assertIn("matmul_scale_mantissa_memory", rtl)
        self.assertIn("matmul_scale_exponent_memory", rtl)

    def test_parallel2_converter_writes_packed_block_rom(self) -> None:
        from mnist_fc.convert_fc import main
        from unittest.mock import patch

        with TemporaryDirectory() as output_dir:
            with patch(
                "sys.argv",
                [
                    "convert_fc.py",
                    "--parallel-macs",
                    "2",
                    "--output-dir",
                    output_dir,
                ],
            ):
                main()
            rtl = (Path(output_dir) / "mnist_fc_parallel2.v").read_text()
            memory_lines = (
                Path(output_dir) / "pair_weight_memory.mem"
            ).read_text().splitlines()

        self.assertIn(
            '(* rom_style = "block", ram_style = "block" *)\n'
            'reg [15:0]  pair_weight_memory [0:1280-1];',
            rtl,
        )
        self.assertIn(
            '$readmemh("pair_weight_memory.mem", pair_weight_memory);',
            rtl,
        )
        self.assertEqual(len(memory_lines), self.layer.input_size * 5)
        self.assertTrue(all(len(line) == 4 for line in memory_lines))


if __name__ == "__main__":
    unittest.main()
