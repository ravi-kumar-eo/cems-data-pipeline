#!/usr/bin/env python3
"""
Script 3: Download GEE Exports from Google Drive

Downloads all activation export folders (named EMSR*) from the root of your
Google Drive into data/GEE_exports/{folder_name}/ using the Google Drive API.

GEE exports each activation as a separate root-level folder in Drive because
toDrive() does not support nested folder paths — this is a GEE limitation.
This script finds all those EMSR* folders automatically and organises them
locally into the flat structure the rest of the pipeline expects.

One-time setup (same Google Cloud project you use for GEE):
  1. Go to console.cloud.google.com → APIs & Services → Enable APIs
     → search "Google Drive API" → Enable
  2. Go to APIs & Services → Credentials → Create Credentials
     → OAuth client ID → Desktop app → Download as credentials.json
  3. Place credentials.json in the repo root (it is gitignored)
  First run will open a browser for you to approve Drive read access.
  Token is saved to data/.gdrive_token.json for all subsequent runs.

Usage:
    python 3_download_gee_exports.py            # download missing folders only
    python 3_download_gee_exports.py --dry-run  # list what would be downloaded
    python 3_download_gee_exports.py --force    # re-download all files
"""

import argparse
import io
import sys
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FOLDER_PREFIX    = "EMSR"          # download Drive root folders starting with this
PARALLEL_WORKERS = 4               # concurrent file downloads per folder
# ─────────────────────────────────────────────────────────────────────────────

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:
    print("ERROR: Google API libraries not found.")
    print("Install with: pip install google-auth-oauthlib google-api-python-client")
    sys.exit(1)

# Drive read-only scope — we never write to the user's Drive
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

BASE_DIR          = Path(__file__).resolve().parent.parent
DATA_DIR          = BASE_DIR / "data"
GEE_EXPORTS_DIR   = DATA_DIR / "GEE_exports"
TOKEN_FILE        = DATA_DIR / ".gdrive_token.json"
CREDENTIALS_FILE  = BASE_DIR / "credentials.json"


# ─── AUTHENTICATION ───────────────────────────────────────────────────────────

def get_drive_service():
    """
    Authenticate with Google Drive API.
    Uses saved token if available, otherwise runs browser OAuth flow.
    """
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print(f"ERROR: credentials.json not found at {CREDENTIALS_FILE}")
                print()
                print("One-time setup:")
                print("  1. Go to console.cloud.google.com → APIs & Services → Credentials")
                print("  2. Create OAuth client ID → Desktop app → Download JSON")
                print("  3. Save it as credentials.json in the repo root")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json())
        print(f"  Token saved to {TOKEN_FILE}")

    return build("drive", "v3", credentials=creds)


# ─── DRIVE HELPERS ───────────────────────────────────────────────────────────

def list_emsr_folders(service, prefix: str) -> list:
    """Return list of (folder_id, folder_name) for root Drive folders matching prefix."""
    folders = []
    page_token = None
    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and name contains '{prefix}' "
        f"and 'root' in parents "
        f"and trashed=false"
    )
    while True:
        resp = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        for f in resp.get("files", []):
            if f["name"].startswith(prefix):
                folders.append((f["id"], f["name"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return sorted(folders, key=lambda x: x[1])


def list_files_in_folder(service, folder_id: str) -> list:
    """Return list of (file_id, file_name) for all files directly in a Drive folder."""
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false and mimeType != 'application/vnd.google-apps.folder'"
    while True:
        resp = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, size)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(service, file_id: str, dest_path: Path) -> bool:
    """Download a single file from Drive to dest_path. Returns True on success."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = service.files().get_media(fileId=file_id)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=64 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return True
    except Exception as e:
        print(f"        ! Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


# ─── MAIN LOGIC ──────────────────────────────────────────────────────────────

def download_folder(service, folder_id: str, folder_name: str,
                    dry_run: bool, force: bool) -> tuple:
    """
    Download all files from one Drive activation folder into
    data/GEE_exports/{folder_name}/.

    Returns (downloaded, skipped, failed) counts.
    """
    local_dir = GEE_EXPORTS_DIR / folder_name
    drive_files = list_files_in_folder(service, folder_id)

    if not drive_files:
        print(f"    ! No files found in Drive folder")
        return 0, 0, 0

    downloaded = skipped = failed = 0

    for f in drive_files:
        fname = f["name"]
        dest  = local_dir / fname

        if dest.exists() and not force:
            skipped += 1
            continue

        size_mb = int(f.get("size", 0)) / (1024 * 1024)
        if dry_run:
            print(f"      [dry-run] {fname}  ({size_mb:.1f} MB)")
            downloaded += 1
            continue

        print(f"      {fname}  ({size_mb:.1f} MB) ...", end=" ", flush=True)
        ok = download_file(service, f["id"], dest)
        if ok:
            print("done")
            downloaded += 1
        else:
            failed += 1

    return downloaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download GEE exports from Google Drive"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files that already exist locally")
    args = parser.parse_args()

    print("=" * 72)
    print("  Script 3: Download GEE Exports from Google Drive")
    print(f"  Output dir : {GEE_EXPORTS_DIR}")
    if args.dry_run:
        print("  Mode       : DRY RUN")
    elif args.force:
        print("  Mode       : FORCE (re-download existing)")
    else:
        print("  Mode       : INCREMENTAL (skip existing files)")
    print("=" * 72)

    # Authenticate
    print("\nAuthenticating with Google Drive …")
    service = get_drive_service()
    print("  Authenticated")

    # Find all EMSR* folders in Drive root
    print(f"\nSearching Drive root for folders starting with '{FOLDER_PREFIX}' …")
    folders = list_emsr_folders(service, FOLDER_PREFIX)

    if not folders:
        print(f"  ! No folders found starting with '{FOLDER_PREFIX}' in Drive root.")
        print("  Make sure GEE export tasks have completed (check code.earthengine.google.com → Tasks).")
        return 1

    print(f"  Found {len(folders)} activation folders in Drive")

    GEE_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total_dl = total_skip = total_fail = 0

    for i, (folder_id, folder_name) in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] {folder_name}")
        dl, skip, fail = download_folder(
            service, folder_id, folder_name, args.dry_run, args.force
        )
        total_dl   += dl
        total_skip += skip
        total_fail += fail

        if skip and not args.force:
            print(f"    {skip} file(s) already exist — skipped (use --force to re-download)")

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    if args.dry_run:
        print(f"  Would download : {total_dl} files across {len(folders)} folders")
    else:
        print(f"  Downloaded     : {total_dl}")
        print(f"  Skipped        : {total_skip}")
        print(f"  Failed         : {total_fail}")
        print(f"  Output         : {GEE_EXPORTS_DIR}")
    print("=" * 72)

    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
