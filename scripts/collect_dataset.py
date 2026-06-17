"""
Collect a large osu! replay dataset by searching ranked beatmaps and downloading replays.

Usage:
    py -3.11 scripts/collect_dataset.py --beatmaps 100 --replays 10 --out data/replays
    py -3.11 scripts/collect_dataset.py --stars-min 3 --stars-max 5 --beatmaps 200 --replays 5

Environment:
    OSU_CLIENT_ID, OSU_CLIENT_SECRET  (required)
    OSU_RATE_DELAY                    (optional, default 0.3s)

Progress is saved to <out>/.progress.json so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
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
    body = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "public",
    }).encode()
    req = urllib.request.Request(_TOKEN_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["access_token"]


# ── API helpers ────────────────────────────────────────────────────────────────

def _get(token: str, path: str, params: dict | None = None) -> dict | list:
    url = f"{_API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())  # type: ignore[return-value]


def _get_bytes(url: str, token: str, retries: int = 4) -> bytes:
    """Download bytes with retry on 429 (backoff) and network errors."""
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = int(e.headers.get("Retry-After", 60))
                print(f" [429 rate-limit, waiting {wait}s]", end=" ", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            raise
        except (urllib.error.URLError, OSError) as e:
            if attempt < retries:
                wait = 15 * (attempt + 1)
                print(f" [network error: {e}, retry in {wait}s]", end=" ", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            raise


# ── Beatmap search ─────────────────────────────────────────────────────────────

def search_beatmaps(
    token: str,
    stars_min: float,
    stars_max: float,
    target: int,
    rate_delay: float,
) -> list[int]:
    """Return up to `target` beatmap IDs in the given star range from ranked maps."""
    beatmap_ids: list[int] = []
    cursor: str | None = None
    page = 0

    print(f"Searching ranked osu! beatmaps [{stars_min}-{stars_max} stars], target={target}...")

    while len(beatmap_ids) < target:
        params: dict = {"m": "0", "s": "ranked", "sort": "plays_desc"}
        if cursor:
            params["cursor_string"] = cursor

        try:
            data = _get(token, "/beatmapsets/search", params)
        except urllib.error.HTTPError as e:
            print(f"  Search page {page} failed: {e}")
            break

        beatmapsets = data.get("beatmapsets", [])  # type: ignore[union-attr]
        if not beatmapsets:
            break

        for bms in beatmapsets:
            for bm in bms.get("beatmaps", []):
                if bm.get("mode") != "osu":
                    continue
                stars = bm.get("difficulty_rating", 0.0)
                if stars_min <= stars <= stars_max:
                    bm_id = bm["id"]
                    if bm_id not in beatmap_ids:
                        beatmap_ids.append(bm_id)
                        if len(beatmap_ids) >= target:
                            break
            if len(beatmap_ids) >= target:
                break

        cursor = data.get("cursor_string")  # type: ignore[union-attr]
        page += 1
        print(f"  Page {page}: found {len(beatmap_ids)}/{target} beatmaps so far...")

        if not cursor:
            print("  No more pages.")
            break

        time.sleep(rate_delay)

    print(f"Search complete: {len(beatmap_ids)} beatmaps collected.\n")
    return beatmap_ids[:target]


# ── Replay fetch ───────────────────────────────────────────────────────────────

def fetch_scores(token: str, beatmap_id: int, limit: int) -> list[dict]:
    data = _get(token, f"/beatmaps/{beatmap_id}/scores", {"limit": limit, "mode": "osu"})
    return data.get("scores", [])  # type: ignore[union-attr]


def fetch_osu_file(beatmap_id: int) -> bytes:
    return _get_bytes(f"{_BASE}/osu/{beatmap_id}", token="")


def fetch_replay(token: str, score_id: int) -> bytes:
    return _get_bytes(f"{_API}/scores/osu/{score_id}/download", token=token)


# ── Progress tracking ──────────────────────────────────────────────────────────

def load_progress(out_root: pathlib.Path) -> set[str]:
    path = out_root / ".progress.json"
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_progress(out_root: pathlib.Path, done: set[str]) -> None:
    path = out_root / ".progress.json"
    path.write_text(json.dumps(sorted(done)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect osu! replay dataset for LunImago training"
    )
    parser.add_argument("--beatmaps", type=int, default=100, help="Number of beatmaps to target")
    parser.add_argument("--replays", type=int, default=10, help="Max replays per beatmap")
    parser.add_argument(
        "--stars-min",
        type=float,
        default=2.0,
        help="Min star rating (default 2.0)",
    )
    parser.add_argument(
        "--stars-max",
        type=float,
        default=6.0,
        help="Max star rating (default 6.0)",
    )
    parser.add_argument("--out", default="data/replays", help="Output directory")
    parser.add_argument(
        "--beatmap-ids",
        nargs="*",
        type=int,
        help="Use specific beatmap IDs instead of search",
    )
    args = parser.parse_args()

    client_id = os.environ.get("OSU_CLIENT_ID", "")
    client_secret = os.environ.get("OSU_CLIENT_SECRET", "")
    rate_delay = float(os.environ.get("OSU_RATE_DELAY", "0.3"))

    if not client_id or not client_secret:
        print("ERROR: OSU_CLIENT_ID and OSU_CLIENT_SECRET must be set.", file=sys.stderr)
        sys.exit(1)

    print("Authenticating...")
    token = get_token(client_id, client_secret)
    print("[OK] Token obtained\n")

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    done = load_progress(out_root)
    print(f"Resume: {len(done)} pairs already downloaded.\n")

    # Collect beatmap IDs
    if args.beatmap_ids:
        beatmap_ids = args.beatmap_ids
        print(f"Using {len(beatmap_ids)} provided beatmap IDs.\n")
    else:
        beatmap_ids = search_beatmaps(
            token, args.stars_min, args.stars_max, args.beatmaps, rate_delay
        )

    total_ok = 0
    total_skip = 0
    total_already = len(done)

    for bm_idx, beatmap_id in enumerate(beatmap_ids):
        print(f"[{bm_idx+1:04d}/{len(beatmap_ids):04d}] beatmap {beatmap_id}")

        # Fetch scores
        try:
            scores = fetch_scores(token, beatmap_id, args.replays)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            print(f"  Scores FAIL ({e}) - skip")
            time.sleep(rate_delay)
            continue

        if not scores:
            print("  No scores - skip")
            continue

        # Download beatmap .osu
        try:
            osu_bytes = fetch_osu_file(beatmap_id)
        except Exception as e:
            print(f"  .osu FAIL ({e}) - skip")
            time.sleep(rate_delay)
            continue

        for idx, score in enumerate(scores):
            score_id = score["id"]
            pair_key = f"{beatmap_id}_{score_id}"
            pair_dir = out_root / f"{beatmap_id}_{idx:02d}"

            if pair_key in done:
                total_already += 1
                continue

            if not score.get("replay", False):
                total_skip += 1
                continue

            print(f"  [{idx+1:02d}/{len(scores):02d}] score {score_id}...", end=" ", flush=True)
            time.sleep(rate_delay)  # throttle before every download
            try:
                osr_bytes = fetch_replay(token, score_id)
                pair_dir.mkdir(parents=True, exist_ok=True)
                (pair_dir / "replay.osr").write_bytes(osr_bytes)
                (pair_dir / "beatmap.osu").write_bytes(osu_bytes)
                done.add(pair_key)
                save_progress(out_root, done)
                print("OK")
                total_ok += 1
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                print(f"FAIL ({e})")
                total_skip += 1

        time.sleep(rate_delay * 2)  # extra pause between beatmaps

        print()

    print("=" * 50)
    print("Done.")
    print(f"  New pairs:     {total_ok}")
    print(f"  Already done:  {total_already}")
    print(f"  Skipped:       {total_skip}")
    print(f"  Total in dir:  {len(done)}")
    print()
    if len(done) >= 50:
        print("Next step:")
        print(f"  py -3.11 -m lunimago train --game osu --data {out_root} \\")
        print("    --epochs 30 --batch 256 --window 32 --export model.onnx")
    else:
        print("Tip: aim for 1000+ pairs before training for production quality.")
        print("     Increase --beatmaps or --replays and re-run (resumes automatically).")


if __name__ == "__main__":
    main()
