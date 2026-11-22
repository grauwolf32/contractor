-- Receipt prefix ranges use bytewise ordering under every database locale.
CREATE INDEX audit_artifact_links_receipt_order ON audit_artifact_links (audit_id, logical_key COLLATE "C");

-- A retry may clear an unsettled Task projection; settled history is immutable.
CREATE TRIGGER audit_tasks_protect_settled_stop BEFORE UPDATE OF stop_record ON audit_tasks
FOR EACH ROW WHEN (OLD.state='settled') EXECUTE FUNCTION contractor_protect_immutable_history();
CREATE TRIGGER audit_items_protect_settled_stop BEFORE UPDATE OF stop_record ON audit_execution_items
FOR EACH ROW WHEN (OLD.state='settled') EXECUTE FUNCTION contractor_protect_immutable_history();
