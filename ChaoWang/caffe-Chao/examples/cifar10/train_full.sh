#!/usr/bin/env sh
export PYTHONPATH=$PWD/python:$PWD/examples/cifar10
TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver.prototxt

