-- Raise the generic Artifact payload limit without changing package/tool limits.
ALTER TABLE artifact_blobs
    DROP CONSTRAINT artifact_blobs_check,
    ADD CONSTRAINT artifact_blobs_size_bytes_check CHECK (
        size_bytes = octet_length(payload)
        AND size_bytes >= 0
        AND size_bytes <= 67108864
    );
