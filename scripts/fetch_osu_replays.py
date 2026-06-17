"""
Fetch osu! replays + beatmaps from the osu! API v2 and organise them
into the data layout expected by LunImago:

  data/replays/
    <beatmap_id>_<score_idx>/
      replay.osr
      beatmap.osu

Usage:
    py -3.11 scripts/fetch_osu_replays.py --beatmap-ids 75 1001 2001
    py -3.11 scripts/fetch_osu_replays.py --beatmap-ids 75 --limit 20 --out data/replays

Environment variables (required):
    OSU_CLIENT_ID      — from osu.ppy.sh/home/account/edit → OAuth
    OSU_CLIENT_SECRET  — same

Optional:
    OSU_RATE_DELAY     — seconds between requests (default: 0.3)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_BASE = "https://osu.ppy.sh"
_TOKEN_URL = f"{_BASE}/oauth/token"
_API = f"{_BASE}/api/v2"


# ── Auth ───────────────────────────────────────────────────────────────────────


def get_token(client_id: str, client_secret: str) -> str:
    body = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": "public",
        }
    ).encode()
    req = urllib.request.Request(_TOKEN_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["access_token"]


# ── API helpers ────────────────────────────────────────────────────────────────


def _get_json(token: str, path: str, params: dict | None = None) -> dict:
    url = f"{_API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())  # type: ignore[return-value]


def _get_bytes(url: str, token: str | None = None) -> bytes:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_scores(token: str, beatmap_id: int, limit: int) -> list[dict]:
    data = _get_json(token, f"/beatmaps/{beatmap_id}/scores", {"limit": limit, "mode": "osu"})
    return data.get("scores", [])


def fetch_osu_file(beatmap_id: int) -> bytes:
    """Download raw .osu beatmap file — no auth required."""
    return _get_bytes(f"{_BASE}/osu/{beatmap_id}")


def fetch_replay(token: str, score_id: int, mode: str = "osu") -> bytes:
    """Download .osr replay binary via API v2. mode must match the game mode of the score."""
    return _get_bytes(f"{_API}/scores/{mode}/{score_id}/download", token=token)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download osu! replay+beatmap pairs for LunImago training"
    )
    parser.add_argument(
        "--beatmap-ids",
        nargs="+",
        type=int,
        required=True,
        help="osu! beatmap IDs (not beatmapset IDs)",
    )
    parser.add_argument(
        "--out",
        default="data/replays",
        help="Output root directory (default: data/replays)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max replays per beatmap (default: 10)",
    )
    args = parser.parse_args()

    client_id = os.environ.get("OSU_CLIENT_ID", "")
    client_secret = os.environ.get("OSU_CLIENT_SECRET", "")
    rate_delay = float(os.environ.get("OSU_RATE_DELAY", "0.3"))

    if not client_id or not client_secret:
        print("ERROR: OSU_CLIENT_ID and OSU_CLIENT_SECRET must be set.", file=sys.stderr)
        print("       Create an OAuth app at osu.ppy.sh/home/account/edit", file=sys.stderr)
        sys.exit(1)

    print("Authenticating with osu! API v2...")
    try:
        token = get_token(client_id, client_secret)
    except urllib.error.HTTPError as e:
        print(f"ERROR: Auth failed ({e}). Check your client_id / client_secret.", file=sys.stderr)
        sys.exit(1)
    print("[OK] Token obtained\n")

    out_root = pathlib.Path(args.out)
    total_ok = 0
    total_skip = 0

    for beatmap_id in args.beatmap_ids:
        print(f"[beatmap {beatmap_id}] Fetching top {args.limit} score(s)...")

        try:
            scores = fetch_scores(token, beatmap_id, args.limit)
        except urllib.error.HTTPError as e:
            print(f"  ✗ Scores fetch failed: {e} — skipping beatmap\n")
            continue

        if not scores:
            print("  ✗ No scores found — skipping beatmap\n")
            continue
        print(f"  {len(scores)} score(s) available")

        print("  Downloading beatmap .osu...", end=" ", flush=True)
        try:
            osu_bytes = fetch_osu_file(beatmap_id)
            print("OK")
        except Exception as e:
            print(f"FAIL ({e}) — skipping beatmap\n")
            continue

        time.sleep(rate_delay)

        for idx, score in enumerate(scores):
            score_id = score["id"]
            pair_dir = out_root / f"{beatmap_id}_{idx:02d}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [{idx + 1:02d}/{len(scores):02d}] replay {score_id}...", end=" ", flush=True)
            if not score.get("replay", False):
                print("SKIP (no replay stored)")
                total_skip += 1
                with contextlib.suppress(OSError):
                    pair_dir.rmdir()
                continue
            try:
                osr_bytes = fetch_replay(token, score_id)
                (pair_dir / "replay.osr").write_bytes(osr_bytes)
                (pair_dir / "beatmap.osu").write_bytes(osu_bytes)
                print("OK")
                total_ok += 1
            except urllib.error.HTTPError as e:
                print(f"FAIL ({e})")
                total_skip += 1
                with contextlib.suppress(OSError):
                    pair_dir.rmdir()

            time.sleep(rate_delay)

        print()

    print(f"Done - {total_ok} pair(s) written, {total_skip} skipped.")
    print(f"Data root: {out_root.resolve()}")

    if total_ok > 0:
        print("\nNext step:")
        print(f"  py -3.11 -m lunimago train --game osu --data {out_root} --export model.onnx")


if __name__ == "__main__":
    main()
