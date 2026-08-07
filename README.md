# GPU Memory Stress Test & Bandwidth Benchmark

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" />
  <img alt="GPU" src="https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white" />
  <img alt="Test type" src="https://img.shields.io/badge/test-memory%20%7C%20bandwidth-orange" />
</p>

A utility for stressing GPU memory, optionally verifying data integrity, and measuring CPU ↔ GPU transfer bandwidth.

The tool can fill GPU memory with deterministic random data or zeros, stream it through pinned host buffers, and optionally verify transferred data using a hash. It is suitable for both PCIe bandwidth benchmarking and GPU memory error detection.

---

## Features

- Fills GPU memory with test data
- Measures CPU ↔ GPU transfer speed
- Optional data integrity verification
- Multiple hash modes: `sha256`, `blake2b`, `crc32`, `none`
- Deterministic random or zero-filled data patterns
- Asynchronous pinned-memory transfer pipeline
- Configurable chunk size and number of CPU buffers
- Optional CUDA/DMA warmup before measured passes
- Human-friendly size units: `B`, `KiB`, `MiB`, `GiB`, `TiB`

---

## Requirements

- CUDA-capable GPU
- Python 3.9 or newer
- CUDA-capable Python runtime/environment
- Enough free GPU memory for the selected buffer size
- Enough host RAM for pinned buffers

Pinned host memory usage is approximately:

```text
buffers × chunk-size
