#!/usr/bin/env bash
# Hapus baris DB + clip/thumbnail HARI INI (default) atau tanggal yang diberikan.
# File lain (latency*.csv, *.md, predictions.csv, dst) TIDAK disentuh.
#   bash tools/clear_today.sh            # hari ini
#   bash tools/clear_today.sh 2026-09-01 # tanggal lain
set -euo pipefail
cd "$(dirname "$0")/.."
D="${1:-$(date +%F)}"
NEXT="$(date -j -v+1d -f %F "$D" +%F 2>/dev/null || date -d "$D +1 day" +%F)"

psql -d bms_tracking <<SQL
BEGIN;
DELETE FROM occupancy_events WHERE timestamp::date  = DATE '$D';
DELETE FROM camera_events    WHERE created_at::date = DATE '$D';
DELETE FROM detections       WHERE created_at::date = DATE '$D';
DELETE FROM tracklets        WHERE started_at::date = DATE '$D' OR ended_at::date = DATE '$D';
DELETE FROM persons          WHERE first_seen::date = DATE '$D';
COMMIT;
SQL

find output thumbnails -type f -newermt "$D 00:00:00" ! -newermt "$NEXT 00:00:00" -delete 2>/dev/null || true
echo "clip/thumbnail $D tersisa: $(find output thumbnails -type f -newermt "$D 00:00:00" ! -newermt "$NEXT 00:00:00" 2>/dev/null | wc -l | tr -d ' ')"
