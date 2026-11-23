package runstore

// SQL statements for output_publication.go.

// recordRunOutputPublicationSQL records the outcome of publishing Run output
// $3 to the Project's 'outputs' namespace under the same name; an empty error
// code or message is stored as NULL. ON CONFLICT (run_id, output_name) DO
// NOTHING keeps the first record, which the caller then reads and compares.
// The caller appends the column list and a created=true marker.
// Used by PostgresStore.RecordRunOutputPublication.
var recordRunOutputPublicationSQL = `
INSERT INTO workflow_run_output_publications (
    run_id, project_id, output_name, status,
    source_namespace, source_name, source_revision,
    target_namespace, target_name, target_revision,
    error_code, error_message
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7,
    'outputs', $3, $8,
    NULLIF($9, ''), NULLIF($10, '')
)
ON CONFLICT (run_id, output_name) DO NOTHING
RETURNING `
