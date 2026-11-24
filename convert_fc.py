import argparse
from pathlib import Path

from myhdl import ResetSignal, Signal, intbv

from fc_mydhdl import HardwareConstants, fc_serial
from fc_reference import QuantizedFullyConnected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="Directory for generated Verilog",
    )
    args = parser.parse_args()

    layer = QuantizedFullyConnected.from_onnx("mnist-12-int8.onnx")
    constants = HardwareConstants.from_layer(layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        constants.weights,
        constants.matmul_multipliers,
        constants.add_multiplier,
        constants.add_bias_offsets,
        constants.matmul_output_zero_point,
        constants.output_zero_point,
        constants.input_size,
        constants.output_size,
    )
    dut.convert(hdl="Verilog", path=str(args.output_dir), name="mnist_fc_serial")
    print(f"Generated {args.output_dir / 'mnist_fc_serial.v'}")


if __name__ == "__main__":
    main()
