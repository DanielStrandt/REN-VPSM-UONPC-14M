from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify VPSM v5 UO NPC Final bundle")
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Verify manifest hashes only; do not load tokenizer/checkpoint/model.",
    )
    args = parser.parse_args()

    manifest_path = ROOT / "bundle_manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed = False
    print("=" * 88)
    print(f"VERIFYING: {manifest.get('release_name')} / {manifest.get('release_tag')}")
    print("=" * 88)

    for row in manifest.get("files", []):
        rel = row["path"]
        expected_size = int(row["size_bytes"])
        expected_hash = str(row["sha256"])
        path = ROOT / Path(rel)
        if not path.is_file():
            print(f"MISSING  {rel}")
            failed = True
            continue
        actual_size = int(path.stat().st_size)
        if actual_size != expected_size:
            print(f"SIZE BAD {rel} | {actual_size} != {expected_size}")
            failed = True
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            print(f"HASH BAD {rel}")
            failed = True
            continue
        print(f"OK       {rel}")

    if failed:
        print("\nBUNDLE VERIFY FAILED")
        return 3

    if args.hash_only:
        print("\nHASH VERIFY PASSED")
        return 0

    sys.path.insert(0, str(ROOT))
    try:
        from runtime.vpsm_npc_runtime import (
            CompactEnglishTokenizer,
            DEFAULT_CHECKPOINT,
            DEFAULT_TOKENIZER,
            load_final_model,
        )

        tok = CompactEnglishTokenizer.load(DEFAULT_TOKENIZER)
        actual_fp = tok.fingerprint()
        expected_fp = str(manifest["tokenizer"]["fingerprint"])
        if actual_fp != expected_fp:
            raise RuntimeError(
                "tokenizer fingerprint mismatch: "
                f"{actual_fp} != {expected_fp}"
            )
        print(f"\ntokenizer fingerprint: {actual_fp}")
        print(f"tokenizer vocab:       {tok.vocab_size:,}")

        loaded = load_final_model(DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER, device="cpu")
        expected_epoch = int(manifest["checkpoint"]["epoch"])
        expected_params = int(manifest["checkpoint"]["parameter_count"])
        if loaded.checkpoint_epoch != expected_epoch:
            raise RuntimeError(
                f"checkpoint epoch {loaded.checkpoint_epoch} != {expected_epoch}"
            )
        if loaded.parameter_count != expected_params:
            raise RuntimeError(
                f"parameter count {loaded.parameter_count} != {expected_params}"
            )
        print(f"checkpoint epoch:      {loaded.checkpoint_epoch}")
        print(f"model parameters:      {loaded.parameter_count:,}")
        print("strict model load:     PASS")
    except Exception as exc:
        print(f"\nRUNTIME VERIFY FAILED: {exc}")
        return 4

    print("\nBUNDLE VERIFY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
