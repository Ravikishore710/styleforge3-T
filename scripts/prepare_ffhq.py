"""Prepare FFHQ data.

Modes:
  --source folder --src PATH   copy+resize an existing folder of face images
                               into data/ffhq_r{res} (default working res)
  --source zip    --src PATH   extract a zip of images, then resize
  --source drive  --resolution 128|256|512|1024
                               download the official FFHQ archive via gdown
                               (1024: Google Drive folder, needs gdown)
  --verify                     validate an existing prepared folder

Examples:
  python scripts/prepare_ffhq.py --source folder --src /kaggle/input/ffhq/images --resolution 256
  python scripts/prepare_ffhq.py --source drive --resolution 256
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.data.ffhq import write_manifest


def resize_copy(src: Path, dst: Path, res: int, max_images: int | None):
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no images under {src}")
    if max_images:
        files = files[:max_images]
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        if img.size != (res, res):
            img = img.resize((res, res), Image.LANCZOS)
        img.save(dst / f"{i:05d}.png")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)}")
    print(f"wrote {len(files)} images to {dst}")


def download_official(res: int, dst: Path):
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown not installed: pip install gdown")
    urls = {
        1024: "https://drive.google.com/drive/folders/17LMsARmWhATiSwcRBduMW2aYiQ2pSR2J",
        256: "https://drive.google.com/uc?id=1A3_1B2zG1t0TXMCT6nUGtdNEmYkRQsg3",
        128: "https://drive.google.com/uc?id=1SBJozCTg_9I9NKwHh2euf1bQIcSB8RFd",
    }
    if res not in urls:
        raise SystemExit(f"no direct URL for resolution {res}; use --source folder/zip")
    dst.mkdir(parents=True, exist_ok=True)
    print("downloading FFHQ (this is large; grab a coffee)...")
    gdown.download(urls[res], str(dst / "ffhq.zip"), quiet=False,
                   fuzzy=True, use_cookies=False) if res != 1024 else \
        gdown.download_folder(urls[res], output=str(dst), quiet=False)
    z = dst / "ffhq.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        z.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["folder", "zip", "drive"], default="folder")
    ap.add_argument("--src", default=None)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    dst = Path(args.data_root) / f"ffhq_r{args.resolution}"
    if args.verify:
        info = write_manifest({"data_dir": str(dst)}, str(dst / "manifest.json"))
        print(info)
        return
    if args.source == "folder":
        resize_copy(Path(args.src), dst, args.resolution, args.max_images)
    elif args.source == "zip":
        tmp = dst.parent / "_zip_extract"
        with zipfile.ZipFile(args.src) as zf:
            zf.extractall(tmp)
        resize_copy(tmp, dst, args.resolution, args.max_images)
        shutil.rmtree(tmp, ignore_errors=True)
    elif args.source == "drive":
        download_official(args.resolution, dst)
    info = write_manifest({"data_dir": str(dst)}, str(dst / "manifest.json"))
    print("manifest:", info)


if __name__ == "__main__":
    main()
