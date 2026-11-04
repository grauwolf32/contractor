package postgres

// auditForkMembershipSQL is the one Audit fork membership rule. Both source
// selection statements bind source kind/ID as $1/$2 and target kind/ID as $6/$7.
// IDs come from the caller's trusted claim/Run or authenticated owner context;
// a matching Project alone is insufficient for cross-Audit material access.
const auditForkMembershipSQL = `
      AND ($6::text <> 'audit' OR ($1::text='project' AND EXISTS (
          SELECT 1 FROM audits AS audit JOIN projects AS project ON project.project_id=audit.project_id
          WHERE audit.audit_id=$7 AND audit.project_id=$2
            AND audit.state IN ('draft','active','paused') AND project.lifecycle_state='active'
      )) OR ($1::text='user' AND EXISTS (
          SELECT 1 FROM audits AS audit JOIN projects AS project ON project.project_id=audit.project_id
          WHERE audit.audit_id=$7 AND audit.owner_id=$2 AND project.owner_id=$2
            AND audit.state IN ('draft','active','paused') AND project.lifecycle_state='active'
      )) OR ($1::text='run' AND EXISTS (
          SELECT 1 FROM audits AS audit JOIN workflow_runs AS run ON run.run_id=$2
          WHERE audit.audit_id=$7 AND run.project_id=audit.project_id
            AND EXISTS (SELECT 1 FROM audit_executions AS execution WHERE execution.execution_id=run.audit_execution_id AND execution.audit_id=audit.audit_id)
      )))
      AND ($1::text <> 'audit' OR EXISTS (
          SELECT 1 FROM audits AS audit
          WHERE audit.audit_id=$2 AND (
              ($6::text='project' AND audit.project_id=$7) OR
              ($6::text='run' AND EXISTS (
                  SELECT 1 FROM workflow_runs AS run JOIN audit_executions AS execution
                    ON execution.execution_id=run.audit_execution_id
                  WHERE run.run_id=$7 AND execution.audit_id=audit.audit_id
                    AND run.project_id=audit.project_id
              ))
          )
      ))`
