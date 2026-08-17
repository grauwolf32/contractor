-- Runs once on first cluster init (docker-entrypoint-initdb.d).
-- The pgvector backend also calls CREATE EXTENSION IF NOT EXISTS at runtime,
-- so this is belt-and-suspenders for fresh volumes.
CREATE EXTENSION IF NOT EXISTS vector;
