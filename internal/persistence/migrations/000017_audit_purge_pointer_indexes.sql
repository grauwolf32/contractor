-- Purge clears derived back-references in bounded batches before removing
-- immutable history. Partial indexes keep repeated passes off cleared rows.
CREATE INDEX audit_findings_purge_pointers_idx ON audit_findings (audit_id, finding_id)
WHERE current_assessment_id IS NOT NULL OR current_decision_id IS NOT NULL
   OR duplicate_target_id IS NOT NULL;

CREATE INDEX audit_items_purge_pointer_idx ON audit_items (audit_id, item_id)
WHERE last_execution_item_id IS NOT NULL;
