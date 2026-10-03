-- Polling an accepted submission changes observed_tokens and updated_at, but
-- neither field contributes to the selected member view. Keep those writes for
-- usage accounting and fair Outstanding ordering without invalidating the
-- immutable view projection on every coordinator tick.
DROP TRIGGER eval_submissions_projection ON eval_submissions;

CREATE TRIGGER eval_submissions_projection AFTER INSERT ON eval_submissions
FOR EACH ROW EXECUTE FUNCTION contractor_eval_member_changed();

CREATE TRIGGER eval_submissions_projection_update
AFTER UPDATE OF state, execution_id, execution_project_id, terminal_state ON eval_submissions
FOR EACH ROW WHEN (
    ROW(OLD.state, OLD.execution_id, OLD.execution_project_id, OLD.terminal_state)
    IS DISTINCT FROM ROW(NEW.state, NEW.execution_id, NEW.execution_project_id, NEW.terminal_state)
) EXECUTE FUNCTION contractor_eval_member_changed();
