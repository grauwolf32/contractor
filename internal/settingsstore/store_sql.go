package settingsstore

// SQL statements for store.go.

// updateSchedulerSettingsSQL sets max_concurrent_runs on the singleton
// scheduler_settings row under CAS on revision ($2, passed as text). Revision
// and updated_at advance only when the value changes; an equal value at the
// expected revision returns the current row unchanged. Returns
// max_concurrent_runs, revision (as text) and updated_at; no row means a stale
// revision. Used by PostgresStore.UpdateSchedulerSettings.
var updateSchedulerSettingsSQL = `
WITH changed AS (
    UPDATE scheduler_settings
       SET max_concurrent_runs = $1,
           revision = revision + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE singleton = true
       AND revision = $2::numeric
       AND max_concurrent_runs IS DISTINCT FROM $1
    RETURNING max_concurrent_runs, revision::text, updated_at
), unchanged AS (
    SELECT max_concurrent_runs, revision::text, updated_at
      FROM scheduler_settings
     WHERE singleton = true
       AND revision = $2::numeric
       AND max_concurrent_runs IS NOT DISTINCT FROM $1
)
SELECT max_concurrent_runs, revision, updated_at FROM changed
UNION ALL
SELECT max_concurrent_runs, revision, updated_at FROM unchanged
LIMIT 1`
