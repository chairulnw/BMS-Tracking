-- Geser waktu hasil run sample_0910 ke waktu kejadian ASLI footage.
--
-- Playlist file diproses "sekarang", jadi detections/tracklets/events distempel
-- wall-clock hari ini. Skrip ini menggeser SEMUA baris dari run tsb mundur
-- sehingga baris paling awal jatuh tepat di 2026-09-10 13:02:00 WIB (jam mulai
-- rekaman asli). Jarak antar-event dipertahankan (geser rigid, bukan skala).
--
-- Pakai (isi run_date = tanggal saat kamu menjalankan playlist):
--   psql -d bms_tracking -v run_date=2026-09-11 -f ai-service/tools/shift_sample_time.sql
--
-- Aman diulang? TIDAK — jalankan sekali per run. Kalau salah, tidak ada undo
-- selain me-run ulang sample-nya.

\set target_start '2026-09-10 13:02:00+07'

BEGIN;

-- offset = (detik paling awal run) - (target). Dihitung sekali.
CREATE TEMP TABLE _off ON COMMIT DROP AS
SELECT (MIN(timestamp) - TIMESTAMPTZ :'target_start') AS d
FROM detections
WHERE timestamp::date = DATE :'run_date';

DO $$
BEGIN
  IF (SELECT d FROM _off) IS NULL THEN
    RAISE EXCEPTION 'Tidak ada baris detections pada run_date — cek -v run_date=';
  END IF;
END $$;

UPDATE detections
   SET timestamp = timestamp - (SELECT d FROM _off)
 WHERE timestamp::date = DATE :'run_date';

UPDATE tracklets
   SET started_at = started_at - (SELECT d FROM _off),
       ended_at   = ended_at   - (SELECT d FROM _off)
 WHERE started_at::date = DATE :'run_date';

UPDATE camera_events
   SET timestamp = timestamp - (SELECT d FROM _off)
 WHERE timestamp::date = DATE :'run_date';

UPDATE occupancy_events
   SET timestamp = timestamp - (SELECT d FROM _off)
 WHERE timestamp::date = DATE :'run_date';

UPDATE persons
   SET first_seen = first_seen - (SELECT d FROM _off),
       last_seen  = last_seen  - (SELECT d FROM _off)
 WHERE first_seen::date = DATE :'run_date';

-- Suffix tanggal di label (Unknown #N@camX@YYYYMMDD) → diturunkan dari tanggal
-- baris itu sendiri (WIB), JADI hanya baris yang barusan digeser yang berubah
-- ke @20260910; data run lama tetap dengan tanggalnya masing-masing.
UPDATE persons SET label = regexp_replace(label, '@[0-9]{8}$',
        '@' || to_char(first_seen AT TIME ZONE 'Asia/Jakarta', 'YYYYMMDD'))
 WHERE label ~ '@[0-9]{8}$';
UPDATE camera_events SET person_label = regexp_replace(person_label, '@[0-9]{8}$',
        '@' || to_char(timestamp AT TIME ZONE 'Asia/Jakarta', 'YYYYMMDD'))
 WHERE person_label ~ '@[0-9]{8}$';
UPDATE occupancy_events SET person_label = regexp_replace(person_label, '@[0-9]{8}$',
        '@' || to_char(timestamp AT TIME ZONE 'Asia/Jakarta', 'YYYYMMDD'))
 WHERE person_label ~ '@[0-9]{8}$';

-- Ringkasan hasil
SELECT 'detections'      AS tabel, MIN(timestamp),  MAX(timestamp)  FROM detections      WHERE timestamp::date  = DATE '2026-09-10'
UNION ALL
SELECT 'tracklets',                MIN(started_at), MAX(ended_at)   FROM tracklets       WHERE started_at::date = DATE '2026-09-10'
UNION ALL
SELECT 'camera_events',            MIN(timestamp),  MAX(timestamp)  FROM camera_events   WHERE timestamp::date  = DATE '2026-09-10'
UNION ALL
SELECT 'occupancy_events',         MIN(timestamp),  MAX(timestamp)  FROM occupancy_events WHERE timestamp::date = DATE '2026-09-10';

COMMIT;
