-- An allocation resource summary tells a report that expired under the
-- telemetry retention apart from one that never arrived (V404-002, OPS-R183).
--
-- stage_allocations.execution_report_received records that an execution
-- report of the allocation was stored. The database sets it on every insert
-- into allocation_execution_reports, so no writer can skip it, and nothing
-- sets it back. Allocations whose reports were deleted before this migration
-- cannot be told apart and keep false.
ALTER TABLE stage_allocations ADD COLUMN execution_report_received boolean DEFAULT false NOT NULL;

UPDATE stage_allocations AS allocation
   SET execution_report_received = true
 WHERE EXISTS (SELECT 1 FROM allocation_execution_reports AS report
                WHERE report.allocation_id = allocation.allocation_id);

CREATE FUNCTION contractor_mark_allocation_report_received() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    UPDATE stage_allocations
       SET execution_report_received = true
     WHERE allocation_id = NEW.allocation_id AND NOT execution_report_received;
    RETURN NULL;
END;
$$;

CREATE TRIGGER allocation_execution_reports_mark_received
    AFTER INSERT ON allocation_execution_reports
    FOR EACH ROW EXECUTE FUNCTION contractor_mark_allocation_report_received();
