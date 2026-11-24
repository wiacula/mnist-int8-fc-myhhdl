#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

docker build \
    -f "$PROJECT_ROOT/tests/Dockerfile.rtl-sim" \
    -t mnist-fc-rtl-sim \
    "$PROJECT_ROOT"
docker run --rm \
    -e MYHDL_VPI=/usr/local/lib/myhdl.vpi \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    mnist-fc-rtl-sim \
    python -m unittest -v tests.test_fc_rtl
