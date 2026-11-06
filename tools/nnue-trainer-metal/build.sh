#!/bin/sh
# Build the Metal NNUE trainer (Swift + Metal, no dependencies; MSL is compiled at runtime).
set -e
cd "$(dirname "$0")"
swiftc -O -swift-version 5 -framework Metal main.swift kernels.swift -o nnue-trainer-metal
