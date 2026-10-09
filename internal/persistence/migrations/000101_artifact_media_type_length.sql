-- The shared canonical media type contract bounds stored metadata to 255 ASCII
-- characters. Application writers already enforce this before publication.
ALTER TABLE artifact_versions ADD CONSTRAINT artifact_versions_media_type_length
    CHECK (length(media_type) BETWEEN 3 AND 255);
