# MNIST int8 fully connected layer

This project studies the final 256-to-10 quantized layer from the ONNX Model Zoo
MNIST-12-int8 model using Python/MyHDL.

## Repository layout

- `mnist_fc/` contains the Python reference, MyHDL design, and RTL converter.
- `models/` contains the float and quantized ONNX models.
- `tests/` contains Python tests, the RTL testbench, and its Docker image.
- `scripts/` contains the test runner and Vivado project/synthesis scripts.
- `docs/` and `reports/` contain the project plan and implementation reports.
- `vivado_project/` contains the serial design's Vivado project and sources.
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
memories, and emits 10 quantized outputs. Its integer requantization logic
reproduces the model's float32 product rounding and is tested against ONNX
Runtime, including near-half rounding cases. The compute phase takes 2,600
cycles per image, excluding the 256-cycle input loading phase.

## MNIST test-set evaluation

Run the evaluator on the official 10,000-image MNIST test split:

```bash
python scripts/evaluate_mnist.py --myhdl-samples 10 \
  --json-out reports/mnist_test_set_results.json
```

The script downloads and checksum-validates the test data if it is not already
cached. It evaluates the float and int8 ONNX models and the Python 256-to-10
reference on all selected images. Float-model pixels are divided by 255; the
int8 model receives pixel values from 0 to 255 with scale 1 and zero point 0.
It also cycle-simulates MyHDL P=1 and P=2 on ten evenly spaced real examples
and on examples that exercise float32 rounding edges or differ between the
Python reference and ONNX Runtime. These simulations do not constitute a full
RTL test on all 10,000 images.

On the complete test split, float and int8 top-1 accuracy are 98.90% and
98.92%. The Python layer reference matches the int8 model byte-for-byte on all
10,000 images. The requantizer now reproduces the model's float32 product
rounding using integer logic. The evaluator also identifies and simulates the
near-half rounding case at sample 4951; P=1 and P=2 both match ONNX Runtime on
that sample and ten evenly spaced examples.

The timing figures below and in the implementation report were measured before
this requantizer change. Re-run Vivado synthesis and routed timing before
treating the previous 25 MHz result as valid for the updated hardware.

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

The Vivado project for the serial P=1 design is
`vivado_project/mnist_fc.xpr`. It targets `xc7z020clg400-1` and includes its
generated Verilog and memory files. It specifies the FPGA part only, with no
board preset or pin constraints. To regenerate the source and create or update
the project from the repository root:

```bash
python -m mnist_fc.convert_fc --output-dir vivado_project/src
vivado -mode batch -source scripts/create_vivado_project.tcl
```

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
project has no fixed clock requirement. The previous timing sweep selected a
25 MHz target (40 ns), but it predates the float32-compatible requantizer and
must be rerun before using it as a timing claim for the current design. See the
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
