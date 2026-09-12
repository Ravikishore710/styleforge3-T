"""Automated checkpoint backup: GitHub Releases & Google Drive.

Runs asynchronously in the background so training is never interrupted.
Uses GitHub Releases API to bypass Git's 100MB file limit for checkpoint archives (up to 2GB).
Automatically detects Kaggle Secrets GITHUB_TOKEN or environment variable.
"""
from __future__ import annotations

import mimetypes
import os
import re
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path

import requests


def get_github_token() -> str | None:
    """Retrieve GitHub token from environment variable or Kaggle Secrets."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    return None


def get_repo_slug(default: str = "Ravikishore710/styleforge3-T") -> str:
    """Detect repo slug (owner/repo) from git remote or return default."""
    try:
        out = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return default


def upload_to_github_release(
    token: str,
    repo_slug: str,
    tag_name: str,
    release_name: str,
    files: list[Path | str],
    body: str = "",
) -> bool:
    """Create or get a GitHub Release and upload file assets."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    api_base = f"https://api.github.com/repos/{repo_slug}"

    try:
        # 1. Get or create release
        rel_url = f"{api_base}/releases/tags/{tag_name}"
        resp = requests.get(rel_url, headers=headers, timeout=30)
        if resp.status_code == 200:
            release = resp.json()
        else:
            payload = {
                "tag_name": tag_name,
                "name": release_name,
                "body": body or f"Automated backup for {tag_name}",
                "draft": False,
                "prerelease": False,
            }
            create_resp = requests.post(
                f"{api_base}/releases", headers=headers, json=payload, timeout=30
            )
            if create_resp.status_code not in (200, 201):
                print(f"[backup] Failed to create release {tag_name}: {create_resp.status_code} {create_resp.text}")
                return False
            release = create_resp.json()

        upload_url_template = release.get("upload_url", "")
        upload_url_base = upload_url_template.split("{")[0]

        # Existing assets map
        existing_assets = {a["name"]: a["id"] for a in release.get("assets", [])}

        # 2. Upload each file
        for f in files:
            fpath = Path(f)
            if not fpath.exists():
                continue
            fname = fpath.name
            # If asset already exists in this release, delete it first to replace cleanly
            if fname in existing_assets:
                del_url = f"{api_base}/releases/assets/{existing_assets[fname]}"
                requests.delete(del_url, headers=headers, timeout=30)

            mime, _ = mimetypes.guess_type(str(fpath))
            content_type = mime or "application/octet-stream"
            up_headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": content_type,
            }
            with open(fpath, "rb") as fp:
                data = fp.read()

            up_url = f"{upload_url_base}?name={fname}"
            up_resp = requests.post(
                up_url, headers=up_headers, data=data, timeout=300
            )
            if up_resp.status_code in (200, 201):
                print(f"[backup] Successfully uploaded {fname} to GitHub Release {tag_name}")
            else:
                print(f"[backup] Failed to upload {fname}: {up_resp.status_code} {up_resp.text}")
        return True
    except Exception as e:
        print(f"[backup] GitHub release upload failed: {e}")
        return False


def create_checkpoint_zip(ckpt_dir: Path, target_zip: Path, prefix: str | None = None) -> Path:
    """Bundle checkpoint files matching prefix (or all current checkpoints) into a zip archive."""
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        meta_file = ckpt_dir / "checkpoint"
        if meta_file.exists():
            zf.write(meta_file, arcname="checkpoint")

        for p in ckpt_dir.iterdir():
            if p.name == "checkpoint" or p.suffix == ".zip":
                continue
            if prefix is None or p.name.startswith(prefix):
                zf.write(p, arcname=p.name)
    return target_zip


def _do_backup(
    output_dir: str,
    kimg: float,
    files_to_upload: list[Path],
    drive_dir: str | None = None,
    github_repo: str | None = None,
):
    """Worker function executed in background thread."""
    try:
        tag_name = f"ckpt-kimg-{int(kimg):04d}"
        release_name = f"Checkpoint kimg {int(kimg)}"
        body = f"StyleForge3-T automated checkpoint snapshot at {kimg:.1f} kimg."

        # Google Drive backup if configured/mounted
        if drive_dir:
            try:
                dpath = Path(drive_dir)
                dpath.mkdir(parents=True, exist_ok=True)
                for f in files_to_upload:
                    if f.exists():
                        shutil.copy2(f, dpath / f.name)
                print(f"[backup] Copied {len(files_to_upload)} files to Drive: {drive_dir}")
            except Exception as e:
                print(f"[backup] Drive copy error: {e}")

        # GitHub Releases backup
        token = get_github_token()
        if token:
            repo = github_repo or get_repo_slug()
            print(f"[backup] Uploading backup to GitHub Releases ({repo}, {tag_name})...")
            upload_to_github_release(
                token=token,
                repo_slug=repo,
                tag_name=tag_name,
                release_name=release_name,
                files=files_to_upload,
                body=body,
            )
        else:
            print("[backup] Note: GITHUB_TOKEN not found in env or Kaggle secrets. Skipping GitHub upload.")
    except Exception as e:
        print(f"[backup] Warning: backup encountered error: {e}")


def trigger_backup(
    output_dir: str,
    kimg: float,
    latest_ckpt_prefix: str | None = None,
    sample_path: Path | None = None,
    drive_dir: str | None = None,
    github_repo: str | None = None,
    async_mode: bool = True,
):
    """Create archive of the latest checkpoint and upload to GitHub and/or Drive."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return

    prefix = None
    if latest_ckpt_prefix:
        prefix = Path(latest_ckpt_prefix).name

    zip_name = f"checkpoint_kimg_{int(kimg):04d}.zip"
    target_zip = ckpt_dir / zip_name

    try:
        create_checkpoint_zip(ckpt_dir, target_zip, prefix=prefix)
    except Exception as e:
        print(f"[backup] Zip creation failed: {e}")
        return

    files_to_upload = [target_zip]
    if sample_path and sample_path.exists():
        files_to_upload.append(sample_path)

    log_path = Path(output_dir) / "logs" / "log.jsonl"
    if log_path.exists():
        files_to_upload.append(log_path)

    if async_mode:
        t = threading.Thread(
            target=_do_backup,
            args=(output_dir, kimg, files_to_upload, drive_dir, github_repo),
            daemon=True,
        )
        t.start()
    else:
        _do_backup(output_dir, kimg, files_to_upload, drive_dir, github_repo)
