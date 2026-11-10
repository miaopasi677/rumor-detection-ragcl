"""Download a large PyTorch wheel using parallel HTTP Range requests.

Existing contiguous prefix bytes in the output file are reused. Each chunk is
stored separately and can be resumed. The final file is assembled in byte order
and verified against the SHA-256 from the official PyTorch wheel index.
"""

import argparse
import hashlib
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


URL = "https://mirrors.aliyun.com/pytorch-wheels/cu113/torch-1.12.0%2Bcu113-cp310-cp310-win_amd64.whl"
SIZE = 2143782114
SHA256 = "0ac18d1d8511533270f38e62243a0ab6aed9863d4b89ddedc1ef5fec87a0fb48"


def download_chunk(index, start, end, part, attempts=6):
    expected = end - start + 1
    temp = part.with_suffix(".tmp")
    for attempt in range(1, attempts + 1):
        current = part.stat().st_size if part.exists() else 0
        if current == expected:
            return index, expected, "complete"
        if current > expected:
            part.unlink()
            current = 0
        range_start = start + current
        temp.unlink(missing_ok=True)
        cmd = [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--range",
            f"{range_start}-{end}",
            "--output",
            str(temp),
            URL,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and temp.exists():
            with temp.open("rb") as src, part.open("ab") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            temp.unlink(missing_ok=True)
        else:
            print(f"chunk {index:02d} attempt {attempt} failed: {result.stderr.strip()}", flush=True)
        if part.exists() and part.stat().st_size == expected:
            return index, expected, "complete"
    size = part.stat().st_size if part.exists() else 0
    raise RuntimeError(f"chunk {index:02d} incomplete: {size}/{expected} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--sha256", default=SHA256)
    args = parser.parse_args()

    output = args.output.resolve()
    parts_dir = output.parent / (output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = math.ceil(args.size / args.parts)

    # Reuse a contiguous prefix already downloaded by a single-stream curl.
    if output.exists() and not (parts_dir / "part_00.bin").exists():
        first = parts_dir / "part_00.bin"
        remaining = min(output.stat().st_size, chunk_size)
        with output.open("rb") as src, first.open("wb") as dst:
            while remaining > 0:
                block = src.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    break
                dst.write(block)
                remaining -= len(block)
        print(f"Reused {first.stat().st_size} prefix bytes.", flush=True)

    ranges = []
    for index in range(args.parts):
        start = index * chunk_size
        end = min(args.size - 1, (index + 1) * chunk_size - 1)
        if start >= args.size:
            break
        part = parts_dir / f"part_{index:02d}.bin"
        ranges.append((index, start, end, part))

    print(f"Downloading {len(ranges)} chunks with {args.parts} workers.", flush=True)
    with ThreadPoolExecutor(max_workers=args.parts) as pool:
        futures = [pool.submit(download_chunk, *r) for r in ranges]
        for future in as_completed(futures):
            index, size, status = future.result()
            print(f"chunk {index:02d}: {status}, {size} bytes", flush=True)

    digest = hashlib.sha256()
    with output.open("wb") as dst:
        for _, _, _, part in ranges:
            with part.open("rb") as src:
                for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    dst.write(block)
                    digest.update(block)
    actual = digest.hexdigest()
    print(f"Assembled: {output}", flush=True)
    print(f"Size: {output.stat().st_size}", flush=True)
    print(f"SHA-256: {actual}", flush=True)
    if output.stat().st_size != args.size or actual != args.sha256:
        raise RuntimeError("Downloaded wheel failed size or SHA-256 verification")
    print("Verification passed.", flush=True)
    for _, _, _, part in ranges:
        part.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
