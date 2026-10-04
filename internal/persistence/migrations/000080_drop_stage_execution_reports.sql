-- allocation_execution_reports replaced stage_execution_reports in migration
-- 000006 and no code has written or read it since. Dropping the table also
-- drops its immutability trigger; the function served no other table.
DROP TABLE stage_execution_reports;
DROP FUNCTION contractor_protect_execution_report_immutable();
