# MNIST int8 fully connected layer

This project studies the final 256-to-10 quantized layer from the ONNX Model Zoo
MNIST-12-int8 model using Python/MyHDL.

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
python convert_fc.py
```

The generated file is written to `build/mnist_fc_serial.v`. FPGA resource
inference still needs to be checked with Vivado for the assigned Zynq-7000
device.
