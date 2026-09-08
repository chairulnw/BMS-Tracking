-- BMS-IIP backend database schema
-- Database: bms_iip
-- Run: psql bms_iip -f backend/db/schema.sql

CREATE TABLE IF NOT EXISTS persons (
    id                  SERIAL PRIMARY KEY,
    name                TEXT        NOT NULL,
    label               TEXT        NOT NULL,          -- original AI label, e.g. "Unknown #1"
    first_seen          TIMESTAMPTZ,
    last_seen           TIMESTAMPTZ,
    last_camera         TEXT,
    best_thumbnail_url  TEXT,
    is_known            BOOLEAN     NOT NULL DEFAULT FALSE,
    enrollment_date     DATE,
    jabatan             TEXT
);

-- Migration (aman dijalankan berulang):
-- ALTER TABLE persons ADD COLUMN IF NOT EXISTS enrollment_date DATE;
ALTER TABLE persons ADD COLUMN IF NOT EXISTS jabatan TEXT;

CREATE INDEX IF NOT EXISTS idx_persons_label ON persons (label);
CREATE INDEX IF NOT EXISTS idx_persons_last_seen ON persons (last_seen DESC);

CREATE TABLE IF NOT EXISTS detections (
    id            SERIAL PRIMARY KEY,
    person_id     INTEGER     NOT NULL REFERENCES persons (id) ON DELETE CASCADE,
    camera_id     TEXT        NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    confidence    FLOAT       NOT NULL,
    method        TEXT        NOT NULL DEFAULT 'unknown',  -- face / appearance / unknown
    track_id      INTEGER,
    thumbnail_url TEXT
);
ALTER TABLE detections ADD COLUMN IF NOT EXISTS track_id      INTEGER;
ALTER TABLE detections ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;
-- Instrumentasi latency: `timestamp` = jam AI service (event terjadi),
-- `created_at` = jam backend insert row ini — selisihnya = network/backend delay.
ALTER TABLE detections ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_detections_person_id ON detections (person_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp  ON detections (timestamp DESC);

-- ── Camera management ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS camera_groups (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cameras (
    id                 SERIAL PRIMARY KEY,
    camera_id          VARCHAR(50)  UNIQUE,
    name               VARCHAR(100) NOT NULL,
    zone_location       VARCHAR(100),
    rtsp_url           TEXT         NOT NULL,
    floor              VARCHAR(50),
    is_active          BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Migrations (aman dijalankan berulang):
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_name = 'cameras' AND column_name = 'zone_location') THEN
    ALTER TABLE cameras RENAME COLUMN zone_location TO location;
  END IF;
END $$;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS group_id INTEGER REFERENCES camera_groups(id) ON DELETE SET NULL;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS analytics_enabled BOOLEAN NOT NULL DEFAULT TRUE;

-- ── Occupancy / zones (line or polygon) ─────────────────────────────────────────

-- Zona itu sendiri gak punya tipe — tipe (line/polygon) melekat ke tiap
-- GAMBAR (satu baris zone_cameras), karena satu zona bisa dipantau kamera A
-- pakai garis dan kamera B pakai polygon sekaligus.
CREATE TABLE IF NOT EXISTS zones (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    max_capacity  INTEGER,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Geometri disimpan per (zona, kamera) karena tiap kamera punya ruang piksel
-- sendiri. points JSONB:
--   type='line'    -> [{"p1":{"x":,"y":},"p2":{"x":,"y":},"in_sign":1}, ...]  (bisa >1 garis)
--   type='polygon' -> [{"x":,"y":}, ...]  (>=3 titik, satu area per kamera)
CREATE TABLE IF NOT EXISTS zone_cameras (
    id          SERIAL PRIMARY KEY,
    zone_id     INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    camera_id   INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    type        VARCHAR(10) NOT NULL CHECK (type IN ('line', 'polygon')),
    points      JSONB   NOT NULL,
    UNIQUE (zone_id, camera_id)
);

CREATE INDEX IF NOT EXISTS idx_zone_cameras_camera ON zone_cameras (camera_id);

-- Migrasi: pindahkan type dari zones (lama) ke zone_cameras (baru), aman di-re-run.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns
             WHERE table_name = 'zones' AND column_name = 'type') THEN
    ALTER TABLE zone_cameras ADD COLUMN IF NOT EXISTS type VARCHAR(10);
    UPDATE zone_cameras zc SET type = z.type
      FROM zones z WHERE zc.zone_id = z.id AND zc.type IS NULL;
    ALTER TABLE zone_cameras ALTER COLUMN type SET NOT NULL;
    ALTER TABLE zone_cameras ADD CONSTRAINT zone_cameras_type_check
      CHECK (type IN ('line', 'polygon'));
    ALTER TABLE zones DROP COLUMN type;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS occupancy_events (
    id           SERIAL PRIMARY KEY,
    camera_id    VARCHAR(50),
    direction    VARCHAR(3)  NOT NULL CHECK (direction IN ('IN', 'OUT')),
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_kind   VARCHAR(20) NOT NULL DEFAULT 'crossing',  -- 'room_entry' | 'passage' | 'crossing'
    snapshot_url TEXT,
    person_label TEXT,
    track_id     INTEGER
);

-- Migrations (aman dijalankan berulang):
ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS zone_camera_id INTEGER REFERENCES zone_cameras(id) ON DELETE CASCADE;
ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS point_x INTEGER;
ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS point_y INTEGER;

-- Migrasi satu-kali dari skema lama (camera_zones 1:1 + crossing_lines) ke
-- zones/zone_cameras many-to-many. Aman di-re-run: camera_zones sudah gak ada
-- setelah migrasi pertama jalan, jadi blok ini otomatis di-skip berikutnya.
DO $$
DECLARE
  cz          RECORD;
  new_zone_id INT;
  agg_points  JSONB;
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'camera_zones') THEN
    FOR cz IN SELECT * FROM camera_zones LOOP
      SELECT jsonb_agg(jsonb_build_object(
               'p1', jsonb_build_object('x', p1_x, 'y', p1_y),
               'p2', jsonb_build_object('x', p2_x, 'y', p2_y),
               'in_sign', in_sign
             ))
        INTO agg_points
        FROM crossing_lines WHERE camera_id = cz.camera_id;

      IF agg_points IS NULL THEN
        CONTINUE;   -- room ada tapi gak ada garis tergambar, skip
      END IF;

      INSERT INTO zones (name, type) VALUES (cz.room_name, 'line') RETURNING id INTO new_zone_id;

      INSERT INTO zone_cameras (zone_id, camera_id, points)
        SELECT new_zone_id, c.id, agg_points
          FROM cameras c WHERE c.camera_id = cz.camera_id;
    END LOOP;

    ALTER TABLE occupancy_events DROP COLUMN IF EXISTS line_id;
    DROP TABLE camera_zones;
    DROP TABLE crossing_lines;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_occupancy_events_timestamp    ON occupancy_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_occupancy_events_zone_camera  ON occupancy_events (zone_camera_id);
CREATE INDEX IF NOT EXISTS idx_occupancy_events_person       ON occupancy_events (person_label);

-- ── Camera events (dashboard / overview) ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS camera_events (
    id           SERIAL PRIMARY KEY,
    camera_id    VARCHAR(50) NOT NULL,
    event_type   VARCHAR(50) NOT NULL,  -- 'person_detected' | 'zone_entry' | 'camera_offline' | 'camera_online'
    category     VARCHAR(20) NOT NULL DEFAULT 'info',  -- 'critical' | 'warning' | 'info'
    description  TEXT,
    snapshot_url TEXT,
    person_label TEXT,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_camera_events_timestamp ON camera_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_camera_events_camera    ON camera_events (camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_events_category  ON camera_events (category);

-- Alarm ack/unack (plan2/spesifikasi.md Fase 3)
ALTER TABLE camera_events ADD COLUMN IF NOT EXISTS acknowledged    BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE camera_events ADD COLUMN IF NOT EXISTS acknowledged_at TIMESTAMPTZ;
-- Instrumentasi latency: `timestamp` = jam AI service (event terjadi),
-- `created_at` = jam backend insert row ini — selisihnya = network/backend delay.
ALTER TABLE camera_events ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
-- total_ai_ms frame terakhir kamera ini saat event digenerate — approx kasar
-- (per-frame, bukan rata-rata seluruh tracklet), lihat backend_client.py.
ALTER TABLE camera_events ADD COLUMN IF NOT EXISTS ai_latency_ms REAL;

-- ── Tracklets (Fase 2 — asosiasi identitas level-tracklet) ──────────────────

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tracklets (
    id                 BIGSERIAL PRIMARY KEY,
    camera_id          VARCHAR(50) NOT NULL,
    track_id           INTEGER     NOT NULL,
    person_id          INT REFERENCES persons(id) ON DELETE SET NULL,  -- NULL = Unassociated
    started_at         TIMESTAMPTZ NOT NULL,
    ended_at           TIMESTAMPTZ NOT NULL,
    n_detections       INTEGER     NOT NULL,
    best_thumbnail_url TEXT,
    embedding          vector,        -- rata-rata top-K crop terbaik, L2-normalized. Unbounded: dimensinya model-specific (OSNet 512, TransReID 3840, BoT-ResNet50 2048)
    assoc_score        REAL,
    attrs              JSONB,         -- diisi Fase 3 (atribut PAR)
    pos_x              INTEGER,       -- titik kaki (foot point) sampel ber-confidence tertinggi (fallback lama),
    pos_y              INTEGER,       -- ruang piksel kamera ini — dipakai tab Pergerakan, TIDAK butuh zona
    positions          JSONB          -- [[x,y], ...] seluruh titik kaki sepanjang hidup tracklet, urut waktu —
                                       -- inilah yang bikin "garis lintasan" beneran jadi garis, bukan 1 titik
);

ALTER TABLE tracklets ADD COLUMN IF NOT EXISTS pos_x INTEGER;
ALTER TABLE tracklets ADD COLUMN IF NOT EXISTS pos_y INTEGER;
ALTER TABLE tracklets ADD COLUMN IF NOT EXISTS positions JSONB;

-- Embedding Re-ID dimensinya model-specific (OSNet 512, TransReID 3840,
-- BoT-ResNet50 2048). Kolomnya unbounded `vector` supaya ganti model Re-ID
-- tidak perlu ALTER lagi. Migrasi lama vector(3840) -> unbounded: isi
-- di-NULL-kan (dimensi tak bisa dikonversi, galeri dibangun ulang dari nol).
--   Ganti model Re-ID? NULL-kan galeri lama:
--   psql -d bms_tracking -c 'UPDATE tracklets SET embedding = NULL;'
-- ivfflat/hnsw pgvector cuma dukung index <=2000 dim, jadi tidak ada index
-- di embedding — similarity search di /people jalan exact scan.
DROP INDEX IF EXISTS idx_tracklets_embedding;

-- pgvector menyimpan dimensi di atttypmod (-1 = unbounded), bukan di
-- character_maximum_length (selalu NULL untuk vector).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'tracklets'::regclass AND attname = 'embedding'
          AND atttypmod <> -1
    ) THEN
        UPDATE tracklets SET embedding = NULL;
        ALTER TABLE tracklets ALTER COLUMN embedding TYPE vector;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_tracklets_person_started
    ON tracklets (person_id, started_at DESC);
-- ponytail: tanpa index ivfflat/hnsw di embedding (lihat DROP INDEX di atas —
-- pgvector cap index-nya di 2000 dim, TransReID 3840), similarity search di
-- /people jalan exact scan. Cukup buat volume sekarang (ratusan-ribuan baris);
-- kalau tracklets tumbuh jauh lebih besar, opsi: PCA/reduksi dimensi embedding
-- sebelum index, atau upgrade pgvector versi yang dukung dim lebih tinggi.
CREATE INDEX IF NOT EXISTS idx_tracklets_attrs
    ON tracklets USING gin (attrs);

ALTER TABLE detections ADD COLUMN IF NOT EXISTS tracklet_id BIGINT REFERENCES tracklets(id) ON DELETE SET NULL;

-- ── Auth ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50) NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);
