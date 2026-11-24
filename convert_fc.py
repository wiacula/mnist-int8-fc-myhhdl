import argparse
from pathlib import Path

from myhdl import ResetSignal, Signal, intbv

from fc_mydhdl import HardwareConstants, fc_serial
from fc_reference import QuantizedFullyConnected


def _request_block_roms(rtl_path: Path, bank_depth: int) -> None:
    rtl = rtl_path.read_text()
    for bank_name in ("even_weight_memory", "odd_weight_memory"):
        declaration = (
            f"reg signed [7:0]  {bank_name} [0:{bank_depth}-1];"
        )
        if rtl.count(declaration) != 1:
            raise RuntimeError(
                f"Could not uniquely find generated ROM {bank_name!r}; "
                "check the MyHDL converter output"
            )
        declaration_end = rtl.index(declaration) + len(declaration)
        initial_start = rtl.find("initial begin", declaration_end)
        initial_end = rtl.find("\nend", initial_start)
        if initial_start < 0 or initial_end < 0:
            raise RuntimeError(
                f"Could not uniquely find the initializer for ROM {bank_name!r}"
            )
        rtl = rtl.replace(
            rtl[initial_start : initial_end + len("\nend")],
            f'initial begin\n    $readmemh("{bank_name}.mem", {bank_name});\nend',
            1,
        )
        declaration = f"reg signed [7:0]  {bank_name} [0:{bank_depth}-1];"
        rtl = rtl.replace(
            declaration,
            f'(* rom_style = "block" *)\n{declaration}',
            1,
        )
    rtl_path.write_text(rtl)


def _write_memory_file(path: Path, values: tuple[int, ...]) -> None:
    path.write_text("".join(f"{value & 0xFF:02x}\n" for value in values))


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
    even_weights = tuple(
        constants.weights[output_index * constants.input_size + input_index]
        for output_index in range(0, constants.output_size, 2)
        for input_index in range(constants.input_size)
    )
    odd_weights = tuple(
        constants.weights[output_index * constants.input_size + input_index]
        for output_index in range(1, constants.output_size, 2)
        for input_index in range(constants.input_size)
    )
    _write_memory_file(args.output_dir / "even_weight_memory.mem", even_weights)
    _write_memory_file(args.output_dir / "odd_weight_memory.mem", odd_weights)

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
    rtl_path = args.output_dir / "mnist_fc_serial.v"
    _request_block_roms(
        rtl_path,
        ((constants.output_size + 1) // 2) * constants.input_size,
    )
    print(f"Generated {rtl_path}")


if __name__ == "__main__":
    main()
