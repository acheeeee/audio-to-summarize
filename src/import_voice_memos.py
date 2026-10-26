"""Copy Apple Voice Memos into data/recordings/<course>/ based on when they were recorded.

Usage:
    .venv/bin/python src/import_voice_memos.py --dry   # preview only
    .venv/bin/python src/import_voice_memos.py         # copy

Reads the Voice Memos database read-only and copies (never moves) each memo to
data/recordings/<course>/<YYYY-MM-DD HHMM>.m4a. Memos shorter than a minute, memos
that match no rule, and memos already copied are skipped.

The terminal app running this needs Full Disk Access (System Settings ->
Privacy & Security), since macOS protects the Voice Memos folder.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC = Path.home() / "Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings"
DST = Path(__file__).resolve().parent.parent / "data" / "recordings"
EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)  # Core Data timestamps count from here
MIN_SECONDS = 60

# (weekday Mon=0, start hour, end hour, course) -- matched on the recording's start time.
SCHEDULE = [
    (0, 0, 12, "機器學習"),
    (1, 9, 10, "實驗室會議"),
    (1, 10, 13, "機器學習"),
    (1, 13, 16, "影像處理"),
    (4, 0, 12, "資料庫系統"),
]


def course(t: datetime) -> str | None:
    h = t.hour + t.minute / 60
    for wd, start, end, name in SCHEDULE:
        if t.weekday() == wd and start <= h < end:
            return name
    return None


def main() -> int:
    dry = "--dry" in sys.argv
    try:
        con = sqlite3.connect(f"file:{SRC / 'CloudRecordings.db'}?mode=ro", uri=True)
        rows = con.execute("select ZDATE, ZDURATION, ZPATH from ZCLOUDRECORDING order by ZDATE").fetchall()
    except sqlite3.OperationalError:
        print("Cannot read Voice Memos. Grant Full Disk Access to this terminal app in "
              "System Settings -> Privacy & Security, then restart it.", file=sys.stderr)
        return 1
    copied = 0
    for zdate, duration, zpath in rows:
        t = (EPOCH + timedelta(seconds=zdate)).astimezone()
        name = course(t)
        if name is None or duration < MIN_SECONDS:
            continue
        dst = DST / name / f"{t:%Y-%m-%d %H%M}.m4a"
        if dst.exists():
            continue
        print(f"{name}\t{dst.name}\t{duration / 60:.0f} min")
        if not dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / zpath, dst)
        copied += 1
    print(f"{'would copy' if dry else 'copied'} {copied} recording(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
