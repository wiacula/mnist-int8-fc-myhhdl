# MNIST int8 fully connected layer

This project studies the final 256-to-10 quantized layer from the ONNX Model Zoo
MNIST-12-int8 model using Python/MyHDL.

## Repository layout

- `mnist_fc/` contains the Python reference, MyHDL design, and RTL converter.
- `models/` contains the quantized ONNX model.
- `tests/` contains Python tests, the RTL testbench, and its Docker image.
- `scripts/` contains the test runner and Vivado synthesis script.
- `docs/` and `reports/` contain the project plan and synthesis report.
- `build/` is generated output and is not tracked by Git.

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reference checks

Run the software reference tests against the quantized ONNX operators:

```bash
python -m unittest -v
```

The reference layer uses the model's `QLinearMatMul` followed by its
`com.microsoft::QLinearAdd`, including the intermediate quantization step.
The model file is from the [ONNX Model Zoo MNIST model](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
and is provided under its MIT license.

The default MyHDL implementation accepts 256 uint8 activations with a
valid/ready handshake, calculates one MAC per cycle using synchronous
memories, and emits 10 quantized outputs. Its fixed-point requantization is
checked over the full possible accumulator ranges and against the ONNX Runtime
reference. The compute phase takes 2,600 cycles per image, excluding the
256-cycle input loading phase.

Generate Verilog with:

```bash
python -m mnist_fc.convert_fc
```

To generate the two-MAC implementation, which calculates an even/odd output
pair in parallel, run:

```bash
python -m mnist_fc.convert_fc --parallel-macs 2
```

The converter generates `build/mnist_fc_serial.v` by default, or
`build/mnist_fc_parallel2.v` for the two-MAC version. The serial design uses
`even_weight_memory.mem` and `odd_weight_memory.mem`; the parallel design packs
each pair of output weights into `pair_weight_memory.mem`. Copy the relevant
RTL and memory file(s) together when moving a design into Vivado. The
synchronous ROMs are initialized with `$readmemh` and marked for block-RAM
inference.

For batch synthesis targeting the device specified in the project plan
(`xc7z020clg400-1`), run the default serial version with:

```bash
vivado -mode batch -source scripts/vivado_synth.tcl \
  -tclargs build/mnist_fc_serial.v build/utilization.rpt
```

For the two-MAC version, pass its top-level module name as the third argument:

```bash
vivado -mode batch -source scripts/vivado_synth.tcl \
  -tclargs build/mnist_fc_parallel2.v build/utilization_parallel2.rpt mnist_fc_parallel2
```

For a routed timing analysis, provide a clock period in nanoseconds. The
project has no fixed clock requirement. The selected 25 MHz target (40 ns) is
the highest tested frequency for which both variants have at least 5 ns of WNS
and zero TNS. At 35 ns, both variants have zero TNS but less than 5 ns of WNS;
at 100 MHz, neither meets timing. See the
[timing and implementation results](reports/parallel_implementation_results.md)
for the sweep and selection method. For example, analyze the serial version at
25 MHz:

```bash
vivado -mode batch -source scripts/vivado_timing.tcl \
  -tclargs build/mnist_fc_serial.v reports/vivado_timing_serial_40ns.rpt mnist_fc_serial 40.0
```

Run the same script for each architecture and period being compared. The timing
summary, critical paths, and utilization are saved as separate reports. This
block-level analysis does not include board-level input/output delays.

## Test generated Verilog

The RTL testbench is written in MyHDL and co-simulates the generated Verilog
with Icarus Verilog. Docker Desktop is used to provide Icarus and the MyHDL VPI
module on macOS:

```bash
sh scripts/run_rtl_tests.sh
```

The co-simulation checks zero, maximum, and deterministic random input vectors
against the Python reference for both implementations, and inserts output
backpressure to test the valid/ready handshake. On systems with Icarus and the
MyHDL VPI module already installed, run it directly with
`python -m unittest -v tests.test_fc_rtl`.
