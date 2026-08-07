GPU Memory Stress Test & Bandwidth Benchmark. Fills GPU memory, optionally verifies data integrity, and measures CPU<->GPU transfer speed.

options:
  -h, --help            show this help message and exit
  --passes N            number of test passes to run. More passes give a more stable result but take longer. (default:
                        3)
  --size SIZE           buffer size per pass, e.g. 512MiB, 1GiB, 2GiB. Default: use almost all free VRAM minus a small
                        reserve.
  --chunk-size SIZE     transfer chunk size, e.g. 64MiB, 128MiB, 256MiB, 512MiB. Larger chunks often give better PCIe
                        throughput, but use more pinned host memory per buffer. (default: 128MiB)
  --hash {sha256,blake2b,crc32,none}
                        integrity hash mode. sha256: strong but slow; blake2b: often faster than sha256; crc32: very
                        fast but non-cryptographic; none: disable verification for maximum bandwidth. (default:
                        sha256)
  --data {random,zeros}
                        data fill mode. random: deterministic random data for memory stress/integrity; zeros: zero-
                        filled data for near-pure PCIe bandwidth testing. (default: random)
  --buffers N           number of pinned CPU buffers for the async pipeline, clamped to 1..8. More buffers may improve
                        overlap between CPU work and PCIe DMA, but increase host RAM usage. (default: 3)
  --no-warmup           disable preliminary CUDA/DMA warmup before measured passes. By default a short warmup is
                        performed to make the first pass more representative.

Examples:
  # Pure PCIe bandwidth test, no verification:
  python gpu_mem_test2_opt.py --passes 2 --size 2GiB --chunk-size 256MiB --data zeros --hash none

  # Fast memory integrity test:
  python gpu_mem_test2_opt.py --passes 1 --chunk-size 128MiB --data random --hash crc32

  # Full deterministic random + SHA-256 stress/verify:
  python gpu_mem_test2_opt.py --passes 3 --data random --hash sha256

Notes:
  Size values accept B, KiB, MiB, GiB, TiB (e.g. 512MiB, 1GiB).
  For maximum bandwidth use --data zeros --hash none.
  For memory error detection use --data random with a hash.
  Pinned host memory usage is roughly: buffers x chunk-size.
