"""Download checkpoints and license texts listed in tools/sources.json."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "models" / "cache" / "src"
LIC_DIR = ROOT / "models" / "licenses"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "scaleup-tools"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def fetch_checkpoint(entry: dict) -> Path:
    dest = SRC_DIR / entry["file"]
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    if "hf_repo" in entry:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(entry["hf_repo"], entry["hf_file"], local_dir=SRC_DIR)
        Path(got).replace(dest) if Path(got) != dest else None
    else:
        http_get(entry["url"], tmp)
        tmp.replace(dest)
    return dest


def fetch_license(entry: dict) -> None:
    dest = LIC_DIR / f"{entry['id']}.txt"
    if dest.exists():
        return
    tmp = dest.with_suffix(".tmp")
    http_get(entry["license_url"], tmp)
    source = entry.get("url") or f"https://huggingface.co/{entry['hf_repo']}"
    header = f"Model: {entry['id']} — Author: {entry['author']} — Source: {source}\n\n"
    dest.write_text(header + tmp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    tmp.unlink()


def main() -> int:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    LIC_DIR.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "tools" / "sources.json").read_text(encoding="utf-8"))
    failed = 0
    for entry in sources:
        try:
            path = fetch_checkpoint(entry)
            fetch_license(entry)
            print(f"OK {entry['id']} {path.stat().st_size} {sha256(path)}")
        except Exception as e:  # report and continue with the rest
            failed += 1
            print(f"FAIL {entry['id']}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
