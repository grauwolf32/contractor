-- Partial judgments reuse existing immutable Artifact receipts. System stop
-- disposition is bounded owner metadata, so exhausted selected-material quota
-- cannot erase the fact that no conclusion was accepted.
ALTER TABLE audit_tasks ADD COLUMN stop_record jsonb;
ALTER TABLE audit_execution_items ADD COLUMN stop_record jsonb;
ALTER TABLE audit_tasks ADD CONSTRAINT audit_tasks_stop_record_check CHECK (
 stop_record IS NULL OR (state='settled' AND result IS NULL
 AND jsonb_typeof(stop_record)='object' AND octet_length(stop_record::text)<=1048576)
);
ALTER TABLE audit_execution_items DROP CONSTRAINT audit_execution_items_collection_shape;
ALTER TABLE audit_execution_items ADD CONSTRAINT audit_execution_items_collection_shape CHECK (
 (state IN ('submitted','collecting') AND collection_disposition IS NULL
 AND result_ref IS NULL AND result_digest IS NULL AND collected_at IS NULL AND stop_record IS NULL)
 OR (state='settled' AND collection_disposition IS NOT NULL AND collected_at IS NOT NULL
 AND (result_ref IS NULL)=(result_digest IS NULL)
 AND (stop_record IS NULL OR (result_ref IS NULL AND jsonb_typeof(stop_record)='object'
 AND octet_length(stop_record::text)<=1048576)))
);
