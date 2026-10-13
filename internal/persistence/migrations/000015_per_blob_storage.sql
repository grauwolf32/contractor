-- Each blob already records its physical backend and object key. New-write
-- policy is process configuration; retaining an installation-wide claim prevents
-- mixed storage without protecting any additional payload invariant.
DROP TRIGGER artifact_blob_backend_guard ON artifact_blobs;
DROP FUNCTION artifact_blob_backend_guard();
DROP TABLE artifact_blob_settings;
