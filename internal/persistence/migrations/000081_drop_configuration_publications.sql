-- Managed configuration publications recorded metadata here that nothing
-- read. The published manifest in the managed configuration root remains the
-- authoritative record.
DROP TABLE configuration_publications;
DROP FUNCTION contractor_protect_configuration_publication_immutable();
