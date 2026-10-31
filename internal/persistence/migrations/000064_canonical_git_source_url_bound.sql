-- LIM-GIT-23 bounds the original input to 4096 UTF-8 bytes. Canonical
-- provenance may percent-encode those bytes; LIM-GIT-24 budgets three times
-- the input bound. Widen the existing constraint without rewriting rows.
ALTER TABLE artifact_versions
    DROP CONSTRAINT artifact_versions_git_repository_url_check,
    ADD CONSTRAINT artifact_versions_git_repository_url_check
        CHECK (octet_length(git_repository_url) BETWEEN 1 AND 12288);
