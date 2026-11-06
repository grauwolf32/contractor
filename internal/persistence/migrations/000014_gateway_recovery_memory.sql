-- Gateway availability is coordinated by the single active Server process.
-- Restart intentionally resets outage gates and retry windows. Existing Run
-- and Stage lifecycle records remain authoritative for interrupted execution.
DROP TABLE gateway_recovery_waits;
DROP TABLE gateway_recovery_failures;
DROP TABLE gateway_allocation_routes;
DROP TABLE gateway_run_routes;
DROP TABLE gateway_recovery_routes;
