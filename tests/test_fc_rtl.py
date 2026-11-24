import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from myhdl import Cosimulation, ResetSignal, Signal, Simulation, StopSimulation, always, delay, instance, intbv

from mnist_fc.fc_reference import QuantizedFullyConnected


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist-12-int8.onnx"
RTL_PATH = ROOT / "build" / "mnist_fc_serial.v"
WRAPPER_PATH = ROOT / "tests" / "rtl_dut_wrapper.v"
VVP_PATH = ROOT / "build" / "mnist_fc_serial.vvp"
VPI_PATH = Path(os.environ.get("MYHDL_VPI", "myhdl.vpi"))


def _simulator_available() -> bool:
    return (
        shutil.which("iverilog") is not None
        and shutil.which("vvp") is not None
        and VPI_PATH.is_file()
    )


@unittest.skipUnless(_simulator_available(), "Icarus Verilog and MyHDL VPI are required")
class GeneratedRtlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mnist_fc.convert_fc",
                "--output-dir",
                str(RTL_PATH.parent),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-s",
                "tb_mnist_fc_serial",
                "-o",
                str(VVP_PATH),
                str(RTL_PATH),
                str(WRAPPER_PATH),
            ],
            cwd=ROOT,
            check=True,
        )
        cls.layer = QuantizedFullyConnected.from_onnx(MODEL_PATH)

    def _run_rtl(self, activations: np.ndarray) -> list[int]:
        clk = Signal(bool(0))
        reset = ResetSignal(0, active=1, isasync=False)
        input_data = Signal(intbv(0)[8:])
        input_valid = Signal(bool(0))
        input_ready = Signal(bool(0))
        output_data = Signal(intbv(0)[8:])
        output_valid = Signal(bool(0))
        output_ready = Signal(bool(0))
        outputs: list[int] = []
        expected = self.layer.run(activations)

        @always(delay(5))
        def clock_driver():
            clk.next = not clk

        @instance
        def stimulus():
            reset.next = True
            input_valid.next = False
            output_ready.next = False
            yield clk.posedge
            yield delay(1)
            reset.next = False

            for value in activations:
                while True:
                    yield clk.negedge
                    if input_ready:
                        input_data.next = int(value)
                        input_valid.next = True
                        yield clk.posedge
                        break
                    yield clk.posedge
                input_valid.next = False

            for cycle in range(5000):
                yield clk.negedge
                ready = cycle % 4 != 0
                output_ready.next = ready
                if output_valid and ready:
                    outputs.append(int(output_data))
                yield clk.posedge
                if len(outputs) == self.layer.output_size:
                    np.testing.assert_array_equal(outputs, expected)
                    raise StopSimulation

            raise AssertionError("RTL timed out before producing all outputs")

        original_directory = Path.cwd()
        try:
            os.chdir(RTL_PATH.parent)
            cosimulation = Cosimulation(
                f"vvp -m {VPI_PATH} {VVP_PATH}",
                clk=clk,
                reset=reset,
                input_data=input_data,
                input_valid=input_valid,
                input_ready=input_ready,
                output_data=output_data,
                output_valid=output_valid,
                output_ready=output_ready,
            )
            Simulation(cosimulation, clock_driver, stimulus).run()
        finally:
            os.chdir(original_directory)
        return outputs

    def test_generated_rtl_matches_reference_for_boundary_and_random_inputs(self) -> None:
        rng = np.random.default_rng(2026)
        vectors = [
            np.zeros(self.layer.input_size, dtype=np.uint8),
            np.full(self.layer.input_size, 255, dtype=np.uint8),
            rng.integers(0, 256, size=self.layer.input_size, dtype=np.uint8),
        ]
        for index, activations in enumerate(vectors):
            with self.subTest(vector=index):
                self._run_rtl(activations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
