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

## Options

| Option | Description | Default |
|---|---|---:|
| `-h`, `--help` | Show help message and exit. | — |
| `--passes N` | Number of test passes to run. More passes give a more stable result but take longer. | `3` |
| `--size SIZE` | Buffer size per pass, e.g. `512MiB`, `1GiB`, `2GiB`. If omitted, the tool uses almost all free VRAM minus a small reserve. | auto |
| `--chunk-size SIZE` | Transfer chunk size, e.g. `64MiB`, `128MiB`, `256MiB`, `512MiB`. Larger chunks often give better PCIe throughput, but use more pinned host memory per buffer. | `128MiB` |
| `--hash {sha256,blake2b,crc32,none}` | Integrity hash mode. `sha256`: strong but slow; `blake2b`: often faster than `sha256`; `crc32`: very fast but non-cryptographic; `none`: disable verification for maximum bandwidth. | `sha256` |
| `--data {random,zeros}` | Data fill mode. `random`: deterministic random data for memory stress/integrity; `zeros`: zero-filled data for near-pure PCIe bandwidth testing. | `random` |
| `--buffers N` | Number of pinned CPU buffers for the async pipeline, clamped to `1..8`. More buffers may improve overlap between CPU work and PCIe DMA, but increase host RAM usage. | `3` |
| `--no-warmup` | Disable preliminary CUDA/DMA warmup before measured passes. By default, a short warmup is performed to make the first pass more representative. | disabled |


Pinned host memory usage is approximately:

```text
buffers × chunk-size


