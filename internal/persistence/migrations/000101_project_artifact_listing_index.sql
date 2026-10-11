-- Serve the generic ProjectScope listing from an index of exactly the
-- listable bindings (V370-025, ART-R115).
--
-- The listing hides the Server-reserved audit- namespaces (ART-R26). Filtered
-- on the primary key, every page from the start of the scope read and dropped
-- every Audit-managed binding, which an Audit with many items makes the bulk
-- of a Project. A key range on the primary key cannot replace the filter: in
-- a linguistic collation such as en_US the audit- names interleave with
-- names such as audit0, because punctuation is ignored at the first
-- comparison level. LIKE matches characters, not collation order, so this
-- partial index holds exactly the listable rows whatever the database
-- collation. listProjectMetadataSQL repeats its predicate literally so the
-- planner can prove it.

CREATE INDEX artifact_bindings_project_listing_idx
    ON artifact_bindings (scope_id, namespace, name)
    WHERE scope_kind = 'project' AND namespace NOT LIKE 'audit-%';
