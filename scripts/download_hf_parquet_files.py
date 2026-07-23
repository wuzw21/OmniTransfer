#!/usr/bin/env python3
"""Download pinned Hugging Face converted parquet files with SHA checks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover converted parquet shards for one Hugging Face dataset, "
            "pin them to a fixed convert commit, and verify their SHA-256 values."
        )
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--file-manifest", type=Path, default=None)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a 40-character commit SHA")
    if (
        args.max_files < 0
        or args.workers <= 0
        or args.retries <= 0
        or args.timeout <= 0
    ):
        raise SystemExit(
            "file limit must be non-negative and worker/retry values positive"
        )

    if args.file_manifest:
        files = load_file_manifest(
            args.file_manifest,
            repo_id=args.repo_id,
            revision=args.revision,
            config=args.config,
            split=args.split,
            endpoint=args.endpoint,
        )
    else:
        files = discover_parquet_files(
            args.repo_id,
            revision=args.revision,
            config=args.config,
            split=args.split,
            endpoint=args.endpoint,
            timeout=args.timeout,
        )
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit("No matching converted parquet files found")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    downloaded = verified_existing = 0
    failures: list[dict[str, str]] = []

    def download(file: dict[str, object]) -> str:
        destination = output / str(file["filename"])
        expected_sha256 = str(file["sha256"])
        if destination.is_file() and _sha256(destination) == expected_sha256:
            return "verified_existing"
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(1, args.retries + 1):
            digest = hashlib.sha256()
            try:
                request = Request(
                    str(file["url"]),
                    headers={"User-Agent": "omnitransfer-data/1.0"},
                )
                with (
                    urlopen(request, timeout=args.timeout) as response,
                    temporary.open("wb") as handle,
                ):
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
        futures = {executor.submit(download, file): file for file in files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({"filename": str(file["filename"]), "error": repr(exc)})
            else:
                if result == "downloaded":
                    downloaded += 1
                else:
                    verified_existing += 1
            print(
                json.dumps(
                    {
                        "processed": downloaded + verified_existing + len(failures),
                        "total": len(files),
                        "downloaded": downloaded,
                        "verified_existing": verified_existing,
                        "failures": len(failures),
                    }
                ),
                flush=True,
            )
    if failures:
        for failure in failures[:20]:
            print(json.dumps(failure), flush=True)
        raise SystemExit(f"Failed to download {len(failures)} parquet files")
    manifest = {
        "schema_version": "omnitransfer_hf_parquet_release_v1",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "config": args.config,
        "split": args.split,
        "files": files,
    }
    temporary = output / "download_manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output / "download_manifest.json")


def discover_parquet_files(
    repo_id: str,
    *,
    revision: str,
    config: str,
    split: str,
    endpoint: str = "https://huggingface.co",
    timeout: float,
) -> list[dict[str, object]]:
    """Return fixed URLs, sizes, and content hashes for converted parquet shards."""

    metadata_url = "https://datasets-server.huggingface.co/parquet?dataset=" + quote(
        repo_id, safe=""
    )
    with urlopen(Request(metadata_url), timeout=timeout) as response:
        payload = json.load(response)
    discovered = []
    for file in payload.get("parquet_files") or ():
        if file.get("config") != config or file.get("split") != split:
            continue
        filename = str(file.get("filename") or "")
        url = fixed_parquet_url(
            repo_id,
            revision=revision,
            config=config,
            split=split,
            filename=filename,
            endpoint=endpoint,
        )
        sha256, size = _remote_identity(url, revision=revision, timeout=timeout)
        discovered.append(
            {"filename": filename, "url": url, "bytes": size, "sha256": sha256}
        )
    return sorted(discovered, key=lambda item: str(item["filename"]))


def fixed_parquet_url(
    repo_id: str,
    *,
    revision: str,
    config: str,
    split: str,
    filename: str,
    endpoint: str = "https://huggingface.co",
) -> str:
    return (
        f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/"
        f"{quote(config, safe='')}/{quote(split, safe='')}/{quote(filename, safe='')}"
    )


def load_file_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    config: str,
    split: str,
    endpoint: str,
) -> list[dict[str, object]]:
    """Load a locally audited file list for hosts without datasets-server access."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in (
        ("repo_id", repo_id),
        ("revision", revision),
        ("config", config),
        ("split", split),
    ):
        if payload.get(key) != expected:
            raise ValueError(
                f"file manifest {key} does not match the requested release"
            )
    files = []
    for item in payload.get("files") or ():
        filename = str(item.get("filename") or "")
        sha256 = str(item.get("sha256") or "")
        size = int(item.get("bytes") or 0)
        if not filename or not re.fullmatch(r"[0-9a-f]{64}", sha256) or size <= 0:
            raise ValueError("file manifest contains an invalid parquet identity")
        files.append(
            {
                "filename": filename,
                "url": fixed_parquet_url(
                    repo_id,
                    revision=revision,
                    config=config,
                    split=split,
                    filename=filename,
                    endpoint=endpoint,
                ),
                "bytes": size,
                "sha256": sha256,
            }
        )
    return sorted(files, key=lambda item: str(item["filename"]))


def _remote_identity(url: str, *, revision: str, timeout: float) -> tuple[str, int]:
    opener = build_opener(_NoRedirect)
    request = Request(
        url, method="HEAD", headers={"User-Agent": "omnitransfer-data/1.0"}
    )
    try:
        opener.open(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        headers = exc.headers
    else:  # pragma: no cover - Hugging Face redirects large parquet files
        raise RuntimeError("Expected a Hugging Face content redirect")
    resolved_revision = str(headers.get("X-Repo-Commit") or "")
    if resolved_revision != revision:
        raise ValueError(
            f"Resolved revision {resolved_revision!r} does not match {revision!r}"
        )
    sha256 = str(headers.get("X-Linked-ETag") or "").strip('"')
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Hugging Face response omitted a SHA-256 linked ETag")
    size = int(headers.get("X-Linked-Size") or 0)
    if size <= 0:
        raise ValueError("Hugging Face response omitted a positive linked size")
    return sha256, size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
