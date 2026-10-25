package runstore

// auditInputMaterializationReadSQL locks one Audit-managed initializing child
// while the product-level caller owns and validates its live Audit claim. This
// lower repository does not read product-owned Audit tables.
const auditInputMaterializationReadSQL = `
SELECT run.audit_input_materialization
  FROM workflow_runs AS run
 WHERE run.run_id=$1 AND run.publication_mode='audit-managed'
   AND run.state='initializing'
 FOR UPDATE OF run`

// auditInputMaterializationWriteSQL advances the persisted exact closure under
// the locked Run. A completed closure is immutable, identity/deadline cannot
// change, and the existing Skill scheduler fence is released only on completion.
// History holders are maintained by the column-scoped migration trigger.
const auditInputMaterializationWriteSQL = `
UPDATE workflow_runs AS run
   SET audit_input_materialization=$3::text::json,
       state_reason_code=CASE WHEN $5 AND skill_snapshot<>'[]'::jsonb THEN $8 ELSE $7 END,
       state_reason_message='', updated_at=clock_timestamp()
 WHERE run.run_id=$1 AND run.publication_mode='audit-managed' AND run.state='initializing'
   AND COALESCE((run.audit_input_materialization->>'revision')::bigint,0)=$2
   AND (run.audit_input_materialization IS NULL OR
       (run.audit_input_materialization->>'identity'=$4
        AND (run.audit_input_materialization->>'complete')::boolean=false
        AND (run.audit_input_materialization->>'deadlineAt')::timestamptz=$6
        AND ($3::text::jsonb->'refs') @> (run.audit_input_materialization::jsonb->'refs')))`
