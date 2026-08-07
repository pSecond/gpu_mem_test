#!/usr/bin/env python3
"""
Second's GPU Bench/test/stress/integrity
By default runs 3 pass error-test of all available GPU memory
"""

import argparse
import hashlib
import re
import sys
import time
import zlib
from collections import deque

try:
    import msvcrt  # type: ignore
except ImportError:
    msvcrt = None

try:
    import torch
except ImportError:
    print("Error: PyTorch with CUDA required. Install via `pip install torch`.")
    sys.exit(1)


INT_MIN = torch.iinfo(torch.int32).min
INT_MAX_EXCLUSIVE = torch.iinfo(torch.int32).max + 1


def parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes. Supports B, KiB, MiB, GiB, TiB."""
    size_str = size_str.strip()
    match = re.match(r'^(\d+)\s*(B|KiB?|MiB?|GiB?|TiB?)?$', size_str, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid size format: '{size_str}'. Use e.g. 128MiB, 1GiB, 512Mi"
        )

    num = int(match.group(1))
    unit = (match.group(2) or 'B').upper().rstrip('B')
    multipliers = {
        '': 1,
        'KI': 1024,
        'MI': 1024 ** 2,
        'GI': 1024 ** 3,
        'TI': 1024 ** 4,
    }

    if unit not in multipliers:
        raise argparse.ArgumentTypeError(
            f"Invalid size unit: '{size_str}'. Use B, KiB, MiB, GiB, TiB."
        )

    return num * multipliers[unit]


class Crc32Hasher:
    """Fast non-cryptographic checksum for memory integrity checks."""

    def __init__(self):
        self.value = 0

    def update(self, data):
        self.value = zlib.crc32(data, self.value) & 0xFFFFFFFF

    def hexdigest(self):
        return f"{self.value:08x}"


def make_hasher(name: str):
    if name == "none":
        return None
    if name == "crc32":
        return Crc32Hasher()
    if name == "blake2b":
        return hashlib.blake2b(digest_size=32)
    return hashlib.sha256()


def hash_update(h, tensor: torch.Tensor):
    """
    Update hasher directly from a CPU tensor buffer.
    Avoids an extra .tobytes() copy.
    """
    if h is None:
        return
    h.update(memoryview(tensor.numpy()))


def make_pinned(length: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(length, dtype=dtype).pin_memory()


def warmup():
    """
    Warm up CUDA context, allocator and DMA path.
    This makes the first measured pass more representative.
    """
    try:
        elems = 8 * 1024 * 1024  # 32 MiB
        gpu = torch.empty(elems, dtype=torch.float32, device="cuda")
        cpu = torch.empty(elems, dtype=torch.float32).pin_memory()

        gpu.copy_(cpu, non_blocking=True)
        torch.cuda.synchronize()

        cpu.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize()

        del gpu, cpu
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_pass(
    buf_bytes: int,
    chunk_bytes: int,
    seed: int,
    hash_name: str,
    buffers: int,
    data_mode: str,
) -> dict:
    res = {"w_spd": 0.0, "r_spd": 0.0, "ok": False, "err": ""}

    n = buf_bytes // 4
    if n == 0:
        res["err"] = "Buffer too small"
        return res

    actual_bytes = n * 4
    chunk_n = max(1, min(chunk_bytes // 4, n))
    num_chunks = (n + chunk_n - 1) // chunk_n
    nb = min(max(1, buffers), num_chunks)

    t_gpu = None
    write_buffers = []
    read_buffers = []

    try:
        t_gpu = torch.empty(n, device="cuda", dtype=torch.float32)

        write_buffers = [make_pinned(chunk_n, torch.int32) for _ in range(nb)]

        if data_mode == "zeros":
            for b in write_buffers:
                b.zero_()

        ref_hash = make_hasher(hash_name)
        gen = torch.Generator(device="cpu")
        stream = torch.cuda.Stream()

        free = deque(range(nb))
        pending = deque()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ================= WRITE PHASE: CPU -> GPU =================
        with torch.cuda.stream(stream):
            for i in range(0, n, chunk_n):
                length = min(chunk_n, n - i)

                if not free:
                    idx, ev = pending.popleft()
                    ev.synchronize()
                    free.append(idx)

                idx = free.popleft()
                buf_i32 = write_buffers[idx][:length]

                if data_mode == "random":
                    gen.manual_seed(seed + i)
                    torch.randint(
                        INT_MIN,
                        INT_MAX_EXCLUSIVE,
                        (length,),
                        generator=gen,
                        dtype=torch.int32,
                        out=buf_i32,
                    )

                cpu_view = buf_i32.view(torch.float32)
                hash_update(ref_hash, cpu_view)

                t_gpu[i:i + length].copy_(cpu_view, non_blocking=True)
                pending.append((idx, stream.record_event()))

            stream.synchronize()

        dt_w = time.perf_counter() - t0
        res["w_spd"] = actual_bytes / dt_w / 1024 ** 2 if dt_w > 0 else 0.0

        # Free write buffers before allocating read buffers.
        del write_buffers
        write_buffers = []

        # ================= READ PHASE: GPU -> CPU =================
        read_buffers = [make_pinned(chunk_n, torch.float32) for _ in range(nb)]
        read_hash = make_hasher(hash_name)

        free = deque(range(nb))
        pending = deque()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.cuda.stream(stream):
            for i in range(0, n, chunk_n):
                length = min(chunk_n, n - i)

                if not free:
                    idx, plen, ev = pending.popleft()
                    ev.synchronize()
                    hash_update(read_hash, read_buffers[idx][:plen])
                    free.append(idx)

                idx = free.popleft()

                read_buffers[idx][:length].copy_(
                    t_gpu[i:i + length],
                    non_blocking=True,
                )
                pending.append((idx, length, stream.record_event()))

            while pending:
                idx, plen, ev = pending.popleft()
                ev.synchronize()
                hash_update(read_hash, read_buffers[idx][:plen])

            stream.synchronize()

        dt_r = time.perf_counter() - t0
        res["r_spd"] = actual_bytes / dt_r / 1024 ** 2 if dt_r > 0 else 0.0

        if ref_hash is None or read_hash is None:
            res["ok"] = True
        else:
            res["ok"] = ref_hash.hexdigest() == read_hash.hexdigest()
            if not res["ok"]:
                res["err"] = "Checksum mismatch!"

        return res

    finally:
        if t_gpu is not None:
            del t_gpu
        if write_buffers:
            del write_buffers
        if read_buffers:
            del read_buffers
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "GPU Memory Stress Test & Bandwidth Benchmark. "
            "Fills GPU memory, optionally verifies data integrity, "
            "and measures CPU<->GPU transfer speed."
        ),
        epilog="""\
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
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--passes",
        type=int,
        default=3,
        metavar="N",
        help=(
            "number of test passes to run. "
            "More passes give a more stable result but take longer. "
            "(default: 3)"
        ),
    )

    ap.add_argument(
        "--size",
        type=str,
        default=None,
        metavar="SIZE",
        help=(
            "buffer size per pass, e.g. 512MiB, 1GiB, 2GiB. "
            "Default: use almost all free VRAM minus a small reserve."
        ),
    )

    ap.add_argument(
        "--chunk-size",
        type=str,
        default="128MiB",
        metavar="SIZE",
        help=(
            "transfer chunk size, e.g. 64MiB, 128MiB, 256MiB, 512MiB. "
            "Larger chunks often give better PCIe throughput, "
            "but use more pinned host memory per buffer. "
            "(default: 128MiB)"
        ),
    )

    ap.add_argument(
        "--hash",
        choices=["sha256", "blake2b", "crc32", "none"],
        default="sha256",
        help=(
            "integrity hash mode. "
            "sha256: strong but slow; "
            "blake2b: often faster than sha256; "
            "crc32: very fast but non-cryptographic; "
            "none: disable verification for maximum bandwidth. "
            "(default: sha256)"
        ),
    )

    ap.add_argument(
        "--data",
        choices=["random", "zeros"],
        default="random",
        help=(
            "data fill mode. "
            "random: deterministic random data for memory stress/integrity; "
            "zeros: zero-filled data for near-pure PCIe bandwidth testing. "
            "(default: random)"
        ),
    )

    ap.add_argument(
        "--buffers",
        type=int,
        default=3,
        metavar="N",
        help=(
            "number of pinned CPU buffers for the async pipeline, clamped to 1..8. "
            "More buffers may improve overlap between CPU work and PCIe DMA, "
            "but increase host RAM usage. "
            "(default: 3)"
        ),
    )

    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help=(
            "disable preliminary CUDA/DMA warmup before measured passes. "
            "By default a short warmup is performed to make the first pass "
            "more representative."
        ),
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
        sys.exit(1)

    if args.passes <= 0:
        print("--passes must be > 0")
        sys.exit(1)

    args.buffers = max(1, min(args.buffers, 8))

    chunk = parse_size(args.chunk_size)

    dev = torch.cuda.current_device()

    if args.size is None:
        free, _ = torch.cuda.mem_get_info(dev)
        buf = max(free - 256 * 1024 ** 2, chunk)
        print(f"Auto: using {buf // 1024 ** 2} MiB of {free // 1024 ** 2} MiB free VRAM")
    else:
        buf = parse_size(args.size)

    if buf < 4:
        print("Buffer too small.")
        sys.exit(1)

    print(
        f"Config: {args.passes} passes × {buf // 1024 ** 2} MiB, "
        f"chunk={chunk // 1024 ** 2} MiB, hash={args.hash}, "
        f"data={args.data}, buffers={args.buffers}"
    )

    if not args.no_warmup:
        warmup()

    errs = 0
    tw = 0.0
    tr = 0.0
    completed = 0

    for p in range(args.passes):
        seed = 42 + p * 1_000_003

        try:
            r = run_pass(
                buf_bytes=buf,
                chunk_bytes=chunk,
                seed=seed,
                hash_name=args.hash,
                buffers=args.buffers,
                data_mode=args.data,
            )
        except Exception as e:
            print(f"Pass {p + 1}/{args.passes}: ✗ FAIL ({e})")
            errs += 1
            continue

        completed += 1

        if r["ok"]:
            if args.hash == "none":
                status = "✓ OK (no verify)"
            else:
                status = "✓ OK"
        else:
            status = f"✗ FAIL ({r['err']})"
            errs += 1

        tw += r["w_spd"]
        tr += r["r_spd"]

        print(
            f"Pass {p + 1}/{args.passes}: "
            f"Write {r['w_spd']:8.1f} MiB/s | "
            f"Read {r['r_spd']:8.1f} MiB/s | "
            f"{status}"
        )

    print("-" * 60)
    print(f"Errors: {errs}/{args.passes}")

    if completed > 0:
        print(f"Avg Write: {tw / completed:.1f} MiB/s")
        print(f"Avg Read:  {tr / completed:.1f} MiB/s")
    else:
        print("No completed passes.")

    if errs == 0:
        print("✅ All passes OK")
    else:
        print("⚠️  MEMORY ERRORS DETECTED")

    if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        if msvcrt:
            print("\nPress any key to exit…")
            msvcrt.getch()
        else:
            input("\nPress Enter to exit…")


if __name__ == "__main__":
    main()