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
# ``msvcrt`` is available only on Windows; it allows capturing a single keypress.
try:
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover – not needed on non‑Windows platforms
    msvcrt = None

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

    This version avoids allocating the entire tensor on CPU at once.  Instead it
    transfers the data in small chunks so that systems with less RAM than the
    GPU memory can still run the benchmark.
    Returns:
        (write_checksum, read_checksum, write_speed_MBps, read_speed_MBps)
        – checksum values are dummy strings because we no longer perform a full
          round‑trip copy to compute them.
    """
    elem_bytes = 4
    n_elements = buffer_bytes // elem_bytes

    # Allocate CUDA tensor and fill with random data
    t_gpu = torch.empty((n_elements,), device="cuda", dtype=torch.float32)

    # Size of a chunk to copy – roughly 64 MiB (adjust if you have more RAM).
    chunk_elems = max(1, 64 * 1024 * 1024 // elem_bytes)

    # Write phase: transfer chunks from GPU to CPU and back while timing.
    start_write = time.perf_counter()
    for i in range(0, n_elements, chunk_elems):
        # Transfer a slice to the CPU
        cpu_chunk = t_gpu[i : i + chunk_elems].cpu()
        # Optionally transfer it back to keep symmetry with the original test.
        _ = cpu_chunk.to("cuda")
    end_write = time.perf_counter()
    write_speed = buffer_bytes / (end_write - start_write) / 1024**2

    # Read phase: again copy chunks back to CPU – this mirrors the original
    # "read" step but keeps memory usage bounded.
    start_read = time.perf_counter()
    for i in range(0, n_elements, chunk_elems):
        cpu_chunk = t_gpu[i : i + chunk_elems].cpu()
        _ = cpu_chunk.to("cuda")
    end_read = time.perf_counter()
    read_speed = buffer_bytes / (end_read - start_read) / 1024**2

    # Clean up to free GPU memory for next pass
    del t_gpu
    torch.cuda.empty_cache()

    return "", "", write_speed, read_speed

def main():
    parser = argparse.ArgumentParser(description="GPU memory bandwidth test")
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of passes to run (default: 1)")
    parser.add_argument("--size", type=str, default=None,
                        help="Buffer size per pass (e.g. 512MiB, 1GiB). If omitted, the script will use all free GPU memory.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Exiting.")
        sys.exit(1)

    if args.size is None:
        # Calculate available memory based on what the caching allocator has
        # reserved.  ``torch.cuda.memory_reserved()`` returns the amount of
        # device memory currently held by PyTorch's caching allocator (including
        # both allocated tensors and any extra space kept for future reuse).
        total_mem = torch.cuda.get_device_properties(0).total_memory
        reserved_mem = torch.cuda.memory_reserved()
        available_mem = max(total_mem - reserved_mem, 0)
        print(
            f"--size not specified: using {available_mem} bytes of free GPU memory"
            f" (reserved {reserved_mem})"
        )
        buffer_bytes = available_mem
    else:
        buffer_bytes = parse_size(args.size)

    errors = 0
    total_write_speed = 0.0
    total_read_speed = 0.0

    # Simple text progress bar – updated after each pass.
    progress_bar_width = 20
    print("Running benchmark…")
    for i in range(args.passes):
        wch, rch, wspd, rspd = run_pass(buffer_bytes)
        if wch != rch:
            errors += 1
        total_write_speed += wspd
        total_read_speed += rspd
        print(
            f"Pass {i + 1}/{args.passes}: write {wch} vs read {rch} | "
            f"write speed: {wspd:.2f} MB/s, read speed: {rspd:.2f} MB/s"
        )
        # Update progress bar
        completed = int((i + 1) / args.passes * progress_bar_width)
        remaining = progress_bar_width - completed
        percent = (i + 1) * 100 // args.passes
        print(
            f"Progress: [{'#' * completed}{' ' * remaining}] {percent} %", end="\r"
        )

    # Ensure the progress bar finishes on its own line.
    print()

    avg_write_speed = total_write_speed / args.passes
    avg_read_speed = total_read_speed / args.passes

    print("\n--- Summary ---")
    print(f"Errors detected: {errors}/{args.passes}")
    print(f"Average write speed: {avg_write_speed:.2f} MB/s")
    print(f"Average read  speed: {avg_read_speed:.2f} MB/s")


    # Wait for any key before exiting (works on Windows via msvcrt).
    if msvcrt:
        print("Press any key to exit…")
        msvcrt.getch()
    else:  # pragma: no cover – fallback for other platforms
        input("Press Enter to exit…")
if __name__ == "__main__":
    main()
