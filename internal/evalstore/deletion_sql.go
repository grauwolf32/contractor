package evalstore

// SQL statements for deletion.go.

// purgeProjectSQL reports whether a Project still has eval state that blocks
// its purge: an experiment not yet fenced by deletion_requested_at, or an
// experiment suboperation still in 'intent'. A true result makes the caller
// return ErrDrain before deleting rows. Used by Store.PurgeProject.
var purgeProjectSQL = `
SELECT EXISTS(SELECT 1
    FROM eval_experiments
    WHERE owner_id=$1
        AND project_id=$2
        AND deletion_requested_at IS NULL) OR EXISTS(SELECT 1
    FROM eval_suboperations op
    JOIN eval_experiments e USING(experiment_id)
    WHERE e.owner_id=$1
        AND e.project_id=$2
        AND op.state='intent')
`
