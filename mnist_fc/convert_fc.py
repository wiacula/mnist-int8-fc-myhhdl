import argparse
from pathlib import Path

from myhdl import ResetSignal, Signal, intbv

from .fc_mydhdl import HardwareConstants, fc_parallel2, fc_serial
from .fc_reference import QuantizedFullyConnected


def _request_block_roms(
    rtl_path: Path,
    memories: tuple[tuple[str, str], ...],
) -> None:
    rtl = rtl_path.read_text()
    for memory_name, declaration in memories:
        if rtl.count(declaration) != 1:
            raise RuntimeError(
                f"Could not uniquely find generated ROM {memory_name!r}; "
                "check the MyHDL converter output"
            )
        declaration_end = rtl.index(declaration) + len(declaration)
        initial_start = rtl.find("initial begin", declaration_end)
        initial_end = rtl.find("\nend", initial_start)
        if initial_start < 0 or initial_end < 0:
            raise RuntimeError(
                f"Could not uniquely find the initializer for ROM {memory_name!r}"
            )
        rtl = rtl.replace(
            rtl[initial_start : initial_end + len("\nend")],
            f'initial begin\n    $readmemh("{memory_name}.mem", {memory_name});\nend',
            1,
        )
        rtl = rtl.replace(
            declaration,
            f'(* rom_style = "block", ram_style = "block" *)\n{declaration}',
            1,
        )
    rtl_path.write_text(rtl)


def _write_memory_file(
    path: Path,
    values: tuple[int, ...],
    hex_digits: int = 2,
) -> None:
    mask = (1 << (hex_digits * 4)) - 1
    path.write_text("".join(f"{value & mask:0{hex_digits}x}\n" for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="Directory for generated Verilog",
    )
    parser.add_argument(
        "--parallel-macs",
        type=int,
        choices=(1, 2),
        default=1,
        help="Number of output MAC units to generate",
    )
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parents[1] / "models" / "mnist-12-int8.onnx"
    layer = QuantizedFullyConnected.from_onnx(model_path)
    constants = HardwareConstants.from_layer(layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parallel_macs == 2:
        memory_name = "pair_weight_memory"
        paired_weights = tuple(
            (
                constants.weights[
                    (output_index * 2) * constants.input_size + input_index
                ]
                & 0xFF
            )
            | (
                (
                    constants.weights[
                        (output_index * 2 + 1) * constants.input_size + input_index
                    ]
                    & 0xFF
                )
                << 8
            )
            for output_index in range(constants.output_size // 2)
            for input_index in range(constants.input_size)
        )
        _write_memory_file(
            args.output_dir / f"{memory_name}.mem",
            paired_weights,
            hex_digits=4,
        )
        memory_specs = (
            (
                memory_name,
                f"reg [15:0]  {memory_name} "
                f"[0:{(constants.output_size // 2) * constants.input_size}-1];",
            ),
        )
    else:
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
        bank_depth = ((constants.output_size + 1) // 2) * constants.input_size
        memory_specs = (
            (
                "even_weight_memory",
                f"reg signed [7:0]  even_weight_memory [0:{bank_depth}-1];",
            ),
            (
                "odd_weight_memory",
                f"reg signed [7:0]  odd_weight_memory "
                f"[0:{bank_depth}-1];",
            ),
        )

    clk = Signal(bool(0))
    reset = ResetSignal(0, active=1, isasync=False)
    input_data = Signal(intbv(0)[8:])
    input_valid = Signal(bool(0))
    input_ready = Signal(bool(0))
    output_data = Signal(intbv(0)[8:])
    output_valid = Signal(bool(0))
    output_ready = Signal(bool(0))

    if args.parallel_macs == 2:
        dut_factory = fc_parallel2
        top_name = "mnist_fc_parallel2"
    else:
        dut_factory = fc_serial
        top_name = "mnist_fc_serial"

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
    dut.convert(hdl="Verilog", path=str(args.output_dir), name=top_name)
    rtl_path = args.output_dir / f"{top_name}.v"
    _request_block_roms(rtl_path, memory_specs)
    print(f"Generated {rtl_path} with {args.parallel_macs} MAC unit(s)")


if __name__ == "__main__":
    main()
