package evalstore

// SQL statements for public_read.go.

// datasetPageSQL returns the Project's eval_collections revision (0 when
// absent) and a JSON array of eval_dataset_revisions metadata after the keyset
// cursor ($3, $4) on (dataset_id, revision), limited to $5.
// Used by Store.DatasetPage.
var datasetPageSQL = `
SELECT COALESCE((SELECT revision
        FROM eval_collections
        WHERE owner_id=$1
            AND project_id=$2), 0), COALESCE((SELECT jsonb_agg(metadata
            ORDER BY dataset_id, revision)
        FROM (SELECT dataset_id, revision, metadata
            FROM eval_dataset_revisions
            WHERE owner_id=$1
                AND project_id=$2
                AND (dataset_id, revision)>($3, $4)
            ORDER BY dataset_id, revision LIMIT $5) page), '[]'::jsonb)
`

// summaryPageSQL returns the Project's collection revision (0 when absent) and
// a JSON array of the owner's experiment summaries, filtered by Project, state,
// dataset and control mode, keyset-paged on (created_at, experiment_id)
// descending after ($6, $7) and limited to $8. Items combine the frozen plan
// (or draft) setup, the published view summary and its freshness against the
// projection queue. Used by Store.SummaryPage.
var summaryPageSQL = `
WITH page AS MATERIALIZED (
    SELECT experiment_id, project_id, name, control_mode, state, revision,
        expected_count, draft, created_at, updated_at, dataset_id
    FROM eval_experiments
    WHERE owner_id = $1
        AND ($2 = '' OR project_id = $2)
        AND ($3 = '' OR state = $3)
        AND ($4 = '' OR dataset_id = $4)
        AND ($5 = '' OR control_mode = $5)
        AND ($6::timestamptz IS NULL OR (created_at, experiment_id) < ($6, $7))
    ORDER BY created_at DESC, experiment_id DESC LIMIT $8
), details AS (
    SELECT page.*, convert_from(COALESCE(p.setup, page.draft), 'UTF8')::jsonb AS setup,
        convert_from(v.summary, 'UTF8')::jsonb AS summary,
        CASE WHEN v.snapshot_id IS NULL THEN 'pending'
             WHEN q.revision = q.published_revision THEN 'current' ELSE 'stale' END AS freshness
    FROM page
    LEFT JOIN eval_frozen_plans p USING (experiment_id)
    LEFT JOIN eval_projection_queue q USING (experiment_id)
    LEFT JOIN eval_view_generations v ON v.experiment_id = page.experiment_id
        AND v.snapshot_id = q.snapshot_id
)
SELECT COALESCE((SELECT revision FROM eval_collections WHERE owner_id = $1 AND project_id = $2), 0),
    COALESCE((SELECT jsonb_agg(jsonb_build_object(
        'experimentId', experiment_id, 'projectId', project_id, 'name', name,
        'controlMode', control_mode, 'state', state, 'revision', revision,
        'expectedMembers', CASE WHEN expected_count > 0 THEN expected_count
            ELSE jsonb_array_length(setup->'caseIds') * (setup->>'repetitions')::int * 2 END,
        'executionKind', setup->'variants'->0->>'kind', 'updatedAt', updated_at,
        'variants', (SELECT jsonb_agg(jsonb_build_object('id', arm->>'id', 'selector', arm->>'selector')
                ORDER BY CASE WHEN arm->>'id' = setup->'comparison'->>'baseline' THEN 0 ELSE 1 END)
            FROM jsonb_array_elements(setup->'variants') arm),
        'datasetId', NULLIF(dataset_id, ''), 'caseCount', jsonb_array_length(setup->'caseIds'),
        'repetitions', (setup->>'repetitions')::int, 'summary', summary, 'freshness', freshness,
        'pageCreatedAt', created_at
    ) ORDER BY created_at DESC, experiment_id DESC) FROM details), '[]'::jsonb)
`
