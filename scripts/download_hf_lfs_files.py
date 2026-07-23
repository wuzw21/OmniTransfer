#!/usr/bin/env python3
"""Download selected Hugging Face LFS files with fixed-revision SHA checks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read LFS pointers from an existing Git checkout, download files "
            "from a pinned Hugging Face revision, and verify every SHA-256."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--file-manifest", type=Path, default=None)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--suffix", action="append", default=[])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--expected-files", type=int, default=0)
    args = parser.parse_args()
    if args.workers <= 0 or args.retries <= 0 or args.timeout <= 0:
        raise SystemExit("workers, retries, and timeout must be positive")
    if args.progress_interval <= 0 or args.max_files < 0 or args.expected_files < 0:
        raise SystemExit("progress interval must be positive and file limits non-negative")

    repository = args.repository.expanduser().resolve()
    if args.file_manifest:
        pointers = load_file_manifest(
            args.file_manifest,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        repository.mkdir(parents=True, exist_ok=True)
    else:
        if not (repository / ".git").exists():
            raise SystemExit(f"Git repository not found: {repository}")
        pointers = load_lfs_pointers(repository)
    pointers = select_lfs_pointers(
        pointers,
        suffixes=tuple(args.suffix),
        paths=tuple(args.path),
    )
    if args.expected_files and len(pointers) != args.expected_files:
        raise SystemExit(
            f"Expected {args.expected_files} selected LFS files, found {len(pointers)}"
        )
    if args.max_files:
        pointers = pointers[: args.max_files]
    if not pointers:
        raise SystemExit("No matching LFS pointers found")

    completed = skipped = 0
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    def download(pointer: tuple[str, str]) -> str:
        expected_sha256, relative_path = pointer
        destination = repository / relative_path
        if destination.is_file() and _sha256(destination) == expected_sha256:
            return "skipped"
        url = (
            f"{args.endpoint.rstrip('/')}/datasets/{args.repo_id}/resolve/"
            f"{args.revision}/{quote(relative_path, safe='/')}"
        )
        temporary = destination.with_suffix(destination.suffix + ".part")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 1):
            digest = hashlib.sha256()
            try:
                request = Request(url, headers={"User-Agent": "omnitransfer-data/1.0"})
                with urlopen(request, timeout=args.timeout) as response, temporary.open(
                    "wb"
                ) as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        digest.update(chunk)
                        handle.write(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError("SHA-256 mismatch")
                temporary.replace(destination)
                return "downloaded"
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == args.retries:
                    raise
                time.sleep(min(2**attempt, 10))
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, pointer): pointer for pointer in pointers}
        for future in as_completed(futures):
            _, relative_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append((relative_path, repr(exc)))
            else:
                if result == "skipped":
                    skipped += 1
                else:
                    completed += 1
            with lock:
                processed = completed + skipped + len(failures)
                if processed % args.progress_interval == 0 or processed == len(pointers):
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "total": len(pointers),
                                "downloaded": completed,
                                "verified_existing": skipped,
                                "failures": len(failures),
                            }
                        ),
                        flush=True,
                    )
    if failures:
        for path, error in failures[:20]:
            print(json.dumps({"failed_path": path, "error": error}), flush=True)
        raise SystemExit(f"Failed to download {len(failures)} LFS files")


def load_lfs_pointers(repository: Path) -> list[tuple[str, str]]:
    """Return ``(sha256, relative_path)`` pairs from ``git lfs ls-files``."""

    result = subprocess.run(
        ["git", "-C", str(repository), "lfs", "ls-files", "-l"],
        check=True,
        capture_output=True,
        text=True,
    )
    pointers: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        match = re.match(r"^([0-9a-f]{64}) [-*] (.+)$", line)
        if match:
            pointers.append((match.group(1), match.group(2)))
    return pointers


def select_lfs_pointers(
    pointers: list[tuple[str, str]],
    *,
    suffixes: tuple[str, ...] = (),
    paths: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Select exact audited paths after optional suffix filtering."""

    selected = list(pointers)
    if suffixes:
        selected = [pointer for pointer in selected if pointer[1].endswith(suffixes)]
    selected_paths = set(paths)
    if selected_paths:
        selected = [pointer for pointer in selected if pointer[1] in selected_paths]
        found_paths = {pointer[1] for pointer in selected}
        missing_paths = sorted(selected_paths - found_paths)
        if missing_paths:
            raise ValueError(f"Selected LFS paths are missing: {missing_paths}")
    return selected


def load_file_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
) -> list[tuple[str, str]]:
    """Load exact LFS identities without cloning a very large Hub repository."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_repo = payload.get("repo_id") or payload.get("dataset")
    manifest_revision = payload.get("revision") or payload.get("source_revision")
    if manifest_repo != repo_id:
        raise ValueError("file manifest repo id does not match the requested dataset")
    if manifest_revision != revision:
        raise ValueError("file manifest revision does not match the requested release")
    files = payload.get("files") or payload.get("selected_files") or ()
    pointers: list[tuple[str, str]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("file manifest entries must be objects")
        relative_path = str(item.get("path") or "")
        sha256 = str(item.get("sha256") or "")
        size = int(item.get("bytes") or 0)
        if (
            not relative_path
            or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            or size <= 0
        ):
            raise ValueError("file manifest contains an invalid LFS identity")
        pointers.append((sha256, relative_path))
    if not pointers:
        raise ValueError("file manifest contains no files")
    return pointers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
