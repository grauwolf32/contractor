-- Operational history is independent of execution truth. Retry cannot extend TTL.
CREATE TABLE performance_minutes (
    server_generation text NOT NULL CHECK (length(server_generation) BETWEEN 1 AND 128),
    minute_start timestamptz NOT NULL CHECK (minute_start = date_trunc('minute', minute_start)),
    schema_version smallint NOT NULL CHECK (schema_version = 1),
    coverage_seconds double precision NOT NULL CHECK (coverage_seconds BETWEEN 0 AND 60),
    payload bytea NOT NULL CHECK (octet_length(payload) BETWEEN 1 AND 32768),
    expires_at timestamptz NOT NULL CHECK (expires_at = minute_start + interval '168 hours'),
    PRIMARY KEY (server_generation, minute_start)
);
CREATE INDEX performance_minutes_range_idx ON performance_minutes (minute_start, server_generation);
CREATE INDEX performance_minutes_expiry_idx ON performance_minutes (expires_at, server_generation, minute_start);
