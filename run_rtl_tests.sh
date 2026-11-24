#!/bin/sh
set -eu

docker build -f Dockerfile.rtl-sim -t mnist-fc-rtl-sim .
docker run --rm \
    -e MYHDL_VPI=/usr/local/lib/myhdl.vpi \
    -v "$PWD:/workspace" \
    -w /workspace \
    mnist-fc-rtl-sim \
    python -m unittest -v test_fc_rtl
