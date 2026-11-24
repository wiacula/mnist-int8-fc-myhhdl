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

The MyHDL implementation accepts 256 uint8 activations with a valid/ready
handshake, calculates one MAC per cycle using synchronous memories, and emits
10 quantized outputs. Its fixed-point requantization is checked over the full
possible accumulator ranges and against the ONNX Runtime reference. The
compute phase takes 2,600 cycles per image, excluding the 256-cycle input
loading phase.

Generate Verilog with:

```bash
python -m mnist_fc.convert_fc
```

The converter generates `build/mnist_fc_serial.v` and two adjacent memory
initialization files, `even_weight_memory.mem` and `odd_weight_memory.mem`.
Copy all three files together when moving the design into Vivado. The weights
are interleaved into even/odd output banks, and both synchronous ROMs are
initialized with `$readmemh` and marked for block-RAM inference.

For batch synthesis targeting `xc7z020clg400-1`, run:

```bash
vivado -mode batch -source scripts/vivado_synth.tcl \
  -tclargs build/mnist_fc_serial.v build/utilization.rpt
```

## Test generated Verilog

The RTL testbench is written in MyHDL and co-simulates the generated Verilog
with Icarus Verilog. Docker Desktop is used to provide Icarus and the MyHDL VPI
module on macOS:

```bash
sh scripts/run_rtl_tests.sh
```

The co-simulation checks zero, maximum, and deterministic random input vectors
against the Python reference, and inserts output backpressure to test the
valid/ready handshake. On systems with Icarus and the MyHDL VPI module already
installed, run it directly with `python -m unittest -v tests.test_fc_rtl`.
