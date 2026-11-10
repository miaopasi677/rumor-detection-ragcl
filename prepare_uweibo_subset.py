import argparse
import hashlib
import json
import random
from pathlib import Path

import py7zr


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Extract a deterministic UWeibo subset for RAGCL word2vec training."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    archive = args.archive.resolve()
    extract_root = args.extract_root.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive, "r") as zf:
        candidates = sorted(
            name
            for name in zf.getnames()
            if name.startswith("UWeibo/dataset/raw/") and name.endswith(".json")
        )
        if len(candidates) < args.count:
            raise RuntimeError(
                f"Archive contains only {len(candidates)} candidates; {args.count} requested."
            )
        selected = sorted(random.Random(args.seed).sample(candidates, args.count))
        print(f"Found {len(candidates)} JSON files; extracting {len(selected)}.", flush=True)
        zf.extract(path=extract_root, targets=selected)

    manifest = {
        "source_archive": str(archive),
        "source_archive_sha256": sha256(archive),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selection": "random.Random(seed).sample(sorted(candidates), count), then sorted",
        "seed": args.seed,
        "files": selected,
    }
    manifest_path = extract_root / "UWeibo" / "uweibo_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
