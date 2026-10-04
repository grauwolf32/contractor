package runstore

// SQL statements for queue_store.go.

// listRunQueueSQL returns one keyset page of an owner's non-terminal Runs,
// oldest first, filtered by optional state and membership (standalone,
// project or evaluation). Rows carry the Project name and kind, the Run event
// cursor (generation, last sequence) and metadata labels as a JSON object.
// Takes no locks. Used by PostgresStore.ListRunQueue.
var listRunQueueSQL = `
WITH page AS (
    SELECT run.run_id, run.project_id, project.name AS project_name,
           project.kind AS project_kind, run.workflow_name,
           run.workflow_version, run.state, run.run_event_generation,
           run.next_run_event_sequence - 1 AS event_sequence,
           run.created_at, run.updated_at
    FROM workflow_runs AS run
    LEFT JOIN projects AS project
      ON project.project_id = run.project_id
     AND project.owner_id = run.owner_id
    WHERE run.owner_id = $1
      AND run.state IN ('initializing', 'pending', 'running', 'waiting', 'cancelling')
      AND ($2::text IS NULL OR run.state = $2)
      AND (
          $3::text IS NULL
          OR ($3 = 'standalone' AND run.project_id IS NULL)
          OR ($3 = 'project' AND project.kind = 'project')
          OR ($3 = 'evaluation' AND project.kind = 'evaluation')
      )
      AND ($4::timestamptz IS NULL OR (run.created_at, run.run_id) > ($4, $5))
    ORDER BY run.created_at, run.run_id
    LIMIT $6
)
SELECT page.run_id, page.project_id, page.project_name, page.project_kind,
       page.workflow_name, page.workflow_version, page.state,
       page.run_event_generation, page.event_sequence,
       page.created_at, page.updated_at,
       COALESCE(
           jsonb_object_agg(labels.label_key, labels.label_value ORDER BY labels.label_key)
               FILTER (WHERE labels.label_key IS NOT NULL),
           '{}'::jsonb
       )
FROM page
LEFT JOIN workflow_run_metadata_labels AS labels USING (run_id)
GROUP BY page.run_id, page.project_id, page.project_name, page.project_kind,
         page.workflow_name, page.workflow_version, page.state,
         page.run_event_generation, page.event_sequence,
         page.created_at, page.updated_at
ORDER BY page.created_at, page.run_id`
