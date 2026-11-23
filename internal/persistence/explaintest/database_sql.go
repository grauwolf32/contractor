package explaintest

// SQL statements for database.go.

// disableAutovacuumSQL turns autovacuum off for every table of the current
// schema. An autovacuum worker's ANALYZE holds a snapshot of its own while it
// samples, which could keep a version the test's VACUUM must remove, and its
// new sample could change the plans between two measurements. Used by
// Database.
const disableAutovacuumSQL = `
DO $$
DECLARE
    relation regclass;
BEGIN
    FOR relation IN
        SELECT oid FROM pg_class WHERE relkind = 'r' AND relnamespace = current_schema()::regnamespace
    LOOP
        EXECUTE format('ALTER TABLE %s SET (autovacuum_enabled = off)', relation);
    END LOOP;
END
$$`
