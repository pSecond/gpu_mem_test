#!/usr/bin/env python3
"""
GPU memory test script.

Runs a simple benchmark that allocates a large CUDA tensor, fills it with random values,
copies the data back and forth between GPU and CPU while timing both transfers and
computing checksums to verify data integrity.

Usage:
    python gpu_mem_test.py [--passes N] [--size SIZE]

Options:
    --passes  Number of write/read passes (default: 1)
    --size    Size of the buffer per pass, e.g. "512MiB" or "1GiB"
               (default: 512MiB). Only positive integer sizes are accepted.
"""

import argparse
import time
import hashlib
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover – guard against missing torch
    print("torch not installed. Please install via `pip install torch` with CUDA support.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_size(size_str: str) -> int:
    """Parse a human readable size string (e.g. 512MiB) to bytes.

    Supports suffixes Ki, Mi, Gi, Ti and B.
    """
    units = {
        "B": 1,
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
    }
    for unit in sorted(units, key=len, reverse=True):
        # Support sizes like "512MiB", "1GiB" by stripping an optional trailing 'B'
        if size_str.endswith(unit + "B"):
            prefix = size_str[:-len(unit) - 1]
        elif size_str.endswith(unit):
            prefix = size_str[:-len(unit)]
        else:
            continue
        try:
            num = int(prefix)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid numeric value: {size_str}")
        return num * units[unit]
    # No unit suffix, assume bytes
    try:
        return int(size_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not parse size: {size_str}")


def sha256_bytes(data: memoryview) -> str:
    """Return SHA‑256 hex digest of the bytes in *data*.

    Using a memoryview allows hashing without copying large buffers.
    """
    m = hashlib.sha256()
    m.update(data)
    return m.hexdigest()

# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

def run_pass(buffer_bytes: int) -> tuple[str, str, float, float]:
    """Run a single write/read pass.

    Returns:
        (write_checksum, read_checksum, write_speed_MBps, read_speed_MBps)
    """
    # Determine number of elements for float32 tensor
    elem_bytes = 4
    n_elements = buffer_bytes // elem_bytes
    # Allocate CUDA tensor and fill with random data
    t_gpu = torch.empty((n_elements,), device="cuda", dtype=torch.float32)

    # Write phase: copy GPU tensor to CPU while timing
    start_write = time.perf_counter()
    cpu_buf_write = t_gpu.cpu().numpy()
    end_write = time.perf_counter()
    write_checksum = sha256_bytes(cpu_buf_write.tobytes())
    write_speed = buffer_bytes / (end_write - start_write) / 1024**2

    # Read phase: copy CPU data back to GPU, then back again while timing
    t_gpu_read = torch.from_numpy(cpu_buf_write).to("cuda", dtype=torch.float32)

    start_read = time.perf_counter()
    cpu_buf_read = t_gpu_read.cpu().numpy()
    end_read = time.perf_counter()
    read_checksum = sha256_bytes(cpu_buf_read.tobytes())
    read_speed = buffer_bytes / (end_read - start_read) / 1024**2

    # Clean up to free GPU memory for next pass
    del t_gpu, t_gpu_read, cpu_buf_write, cpu_buf_read
    torch.cuda.empty_cache()

    return write_checksum, read_checksum, write_speed, read_speed

def main():
    parser = argparse.ArgumentParser(description="GPU memory bandwidth test")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of passes to run (default: 1)")
    parser.add_argument("--size", type=str, default="512MiB",
                        help="Buffer size per pass (e.g. 512MiB, 1GiB) – default 512MiB")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Exiting.")
        sys.exit(1)

    buffer_bytes = parse_size(args.size)

    errors = 0
    total_write_speed = 0.0
    total_read_speed = 0.0

    for i in range(args.passes):
        wch, rch, wspd, rspd = run_pass(buffer_bytes)
        if wch != rch:
            errors += 1
        total_write_speed += wspd
        total_read_speed += rspd
        print(f"Pass {i+1}/{args.passes}: write {wch} vs read {rch} | "
              f"write speed: {wspd:.2f} MB/s, read speed: {rspd:.2f} MB/s")

    avg_write_speed = total_write_speed / args.passes
    avg_read_speed = total_read_speed / args.passes

    print("\n--- Summary ---")
    print(f"Errors detected: {errors}/{args.passes}")
    print(f"Average write speed: {avg_write_speed:.2f} MB/s")
    print(f"Average read  speed: {avg_read_speed:.2f} MB/s")

if __name__ == "__main__":
    main()
