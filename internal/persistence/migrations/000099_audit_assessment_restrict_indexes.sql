-- RESTRICT checks during Audit purge must find references without scanning
-- assessments belonging to every other Audit.
CREATE INDEX audit_finding_assessments_collection_receipt_idx
    ON audit_finding_assessments(collection_receipt_id);
CREATE INDEX audit_finding_assessments_execution_item_idx
    ON audit_finding_assessments(execution_item_id, item_id);
