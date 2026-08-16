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
    enrollment_date     DATE
);

-- Migration (aman dijalankan berulang):
-- ALTER TABLE persons ADD COLUMN IF NOT EXISTS enrollment_date DATE;

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

CREATE INDEX IF NOT EXISTS idx_detections_person_id ON detections (person_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp  ON detections (timestamp DESC);

-- ── Camera management ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cameras (
    id            SERIAL PRIMARY KEY,
    camera_id     VARCHAR(50)  UNIQUE,
    name          VARCHAR(100) NOT NULL,
    zone_location VARCHAR(100),
    rtsp_url      TEXT         NOT NULL,
    floor         VARCHAR(50),
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ── Occupancy / line crossing ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS camera_zones (
    id          SERIAL PRIMARY KEY,
    camera_id   VARCHAR(50)  NOT NULL UNIQUE,
    room_name   VARCHAR(100) NOT NULL,
    floor       VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS crossing_lines (
    id          SERIAL PRIMARY KEY,
    camera_id   VARCHAR(50) NOT NULL,
    p1_x        INT NOT NULL,
    p1_y        INT NOT NULL,
    p2_x        INT NOT NULL,
    p2_y        INT NOT NULL,
    in_sign     INT NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crossing_lines_camera ON crossing_lines (camera_id);

CREATE TABLE IF NOT EXISTS occupancy_events (
    id           SERIAL PRIMARY KEY,
    camera_id    VARCHAR(50),
    line_id      INT REFERENCES crossing_lines (id) ON DELETE CASCADE,
    direction    VARCHAR(3)  NOT NULL CHECK (direction IN ('IN', 'OUT')),
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_kind   VARCHAR(20) NOT NULL DEFAULT 'crossing',  -- 'room_entry' | 'passage' | 'crossing'
    snapshot_url TEXT,
    person_label TEXT,
    track_id     INTEGER
);

-- Migrations (aman dijalankan berulang):
-- ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS event_kind   VARCHAR(20) NOT NULL DEFAULT 'crossing';
-- ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS snapshot_url TEXT;
-- ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS person_label TEXT;
ALTER TABLE occupancy_events ADD COLUMN IF NOT EXISTS track_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_occupancy_events_timestamp ON occupancy_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_occupancy_events_line      ON occupancy_events (line_id);
CREATE INDEX IF NOT EXISTS idx_occupancy_events_person    ON occupancy_events (person_label);

-- ── Camera events (dashboard / overview) ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS camera_events (
    id           SERIAL PRIMARY KEY,
    camera_id    VARCHAR(50) NOT NULL,
    event_type   VARCHAR(50) NOT NULL,  -- 'person_detected' | 'zone_entry' | 'camera_offline' | 'camera_online' | 'after_hours_activity'
    category     VARCHAR(20) NOT NULL DEFAULT 'info',  -- 'critical' | 'warning' | 'info'
    description  TEXT,
    snapshot_url TEXT,
    person_label TEXT,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_camera_events_timestamp ON camera_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_camera_events_camera    ON camera_events (camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_events_category  ON camera_events (category);

-- ── Auth ──────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50) NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);
