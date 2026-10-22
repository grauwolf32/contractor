package pgstore

// correlationFenceSQL reads only the two selected Audit identities and compact
// baseline identities. No profile, evidence or whole-Audit inventory is hydrated.
const correlationFenceSQL = `SELECT project.finding_correlation_revision,
 audit.audit_id,audit.revision,audit.state,audit.profile_name,audit.profile_version,audit.profile_digest,
 CASE WHEN audit.baseline_snapshot IS NULL THEN NULL ELSE 'sha256:'||encode(sha256(convert_to(audit.baseline_snapshot::text,'UTF8')),'hex') END
 FROM projects AS project LEFT JOIN audits AS audit ON audit.project_id=project.project_id
 AND audit.owner_id=$1 AND audit.audit_id=ANY($3::text[])
 WHERE project.owner_id=$1 AND project.project_id=$2 ORDER BY audit.audit_id COLLATE "C"`

// candidateSourceSQL checks the selected receipt through this exact finding's
// first/contributing relation. A foreign receipt is indistinguishable from absent.
const candidateSourceSQL = `SELECT finding.finding_id,finding.audit_id,finding.correlation_project_id,
 finding.first_receipt_id,receipt.receipt_id,finding.correlation_group_key,receipt.comparison_projection,
 audit.revision,project.finding_correlation_revision
 FROM audit_findings AS finding JOIN audits AS audit USING(audit_id)
 JOIN projects AS project ON project.project_id=audit.project_id AND project.owner_id=$1
 JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id=coalesce($4::text,finding.first_receipt_id)
 WHERE audit.owner_id=$1 AND finding.audit_id=$2 AND finding.finding_id=$3
 AND (receipt.receipt_id=finding.first_receipt_id OR EXISTS(
 SELECT 1 FROM audit_finding_contributions AS contribution WHERE contribution.finding_id=finding.finding_id
 AND contribution.audit_id=finding.audit_id AND contribution.receipt_id=receipt.receipt_id))`

// Candidate windows seek first, and inspect at most 898 compact matching
// projections. Nonmatches consume explicit cursor work, never an unbounded sort.
const candidateWindowColumns = `SELECT finding.finding_id,finding.audit_id,finding.first_receipt_id,
 finding.correlation_group_key,finding.created_at,receipt.comparison_projection-'preview' AS comparison_hints
 FROM selected AS finding JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id=finding.first_receipt_id
 ORDER BY finding.created_at,finding.finding_id COLLATE "C"`
const candidateProjectWindowSQL = `WITH selected AS MATERIALIZED (
 SELECT finding_id,audit_id,first_receipt_id,correlation_group_key,created_at FROM audit_findings
 WHERE correlation_project_id=$1 AND finding_id<>$2
 AND (created_at,finding_id COLLATE "C")>($3,$4)
 ORDER BY created_at,finding_id COLLATE "C" LIMIT $5
 ) ` + candidateWindowColumns
const candidateAuditWindowSQL = `WITH selected AS MATERIALIZED (
 SELECT finding_id,audit_id,first_receipt_id,correlation_group_key,created_at FROM audit_findings
 WHERE audit_id=$1 AND finding_id<>$2
 AND (created_at,finding_id COLLATE "C")>($3,$4)
 ORDER BY created_at,finding_id COLLATE "C" LIMIT $5
 ) ` + candidateWindowColumns

// Strong tiers seek their exact equality group directly. Weak tiers retain the
// explicit bounded work window instead of sorting every matching GIN posting.
const candidateProjectGroupWindowSQL = `WITH selected AS MATERIALIZED (
 SELECT finding_id,audit_id,first_receipt_id,correlation_group_key,created_at FROM audit_findings
 WHERE correlation_project_id=$1 AND finding_id<>$2 AND correlation_group_key=$6 AND correlation_group_key<>''
 AND (created_at,finding_id COLLATE "C")>($3,$4)
 ORDER BY created_at,finding_id COLLATE "C" LIMIT $5
 ) ` + candidateWindowColumns
const candidateAuditGroupWindowSQL = `WITH selected AS MATERIALIZED (
 SELECT finding_id,audit_id,first_receipt_id,correlation_group_key,created_at FROM audit_findings
 WHERE audit_id=$1 AND finding_id<>$2 AND correlation_group_key COLLATE "C"=$6 AND correlation_group_key<>''
 AND (created_at,finding_id COLLATE "C")>($3,$4)
 ORDER BY created_at,finding_id COLLATE "C" LIMIT $5
 ) ` + candidateWindowColumns

// comparisonGroupsSQL recursively seeks the next distinct key in each selected
// Audit, skipping a high-multiplicity group's peers with a strict greater-than
// bound. The recursion performs at most limit+1 pairs of index probes.
const comparisonGroupsSQL = `WITH RECURSIVE walk(step,group_key) AS (
 SELECT 0,$3::text COLLATE "C"
 UNION ALL
 SELECT walk.step+1,next_group.key FROM walk
 CROSS JOIN LATERAL (
  SELECT least(l.key,r.key) AS key FROM
  (SELECT (SELECT correlation_group_key COLLATE "C" FROM audit_findings
   WHERE audit_id=$1 AND correlation_group_key COLLATE "C">walk.group_key
   ORDER BY correlation_group_key COLLATE "C",finding_id COLLATE "C" LIMIT 1) AS key) AS l
  CROSS JOIN
  (SELECT (SELECT correlation_group_key COLLATE "C" FROM audit_findings
   WHERE audit_id=$2 AND correlation_group_key COLLATE "C">walk.group_key
   ORDER BY correlation_group_key COLLATE "C",finding_id COLLATE "C" LIMIT 1) AS key) AS r
 ) AS next_group WHERE walk.step<$4 AND next_group.key IS NOT NULL
 )
 SELECT walk.group_key,coalesce(l.correlation_algorithm,r.correlation_algorithm),
 coalesce(l.correlation_value,r.correlation_value),l.finding_id,r.finding_id,''
 FROM walk LEFT JOIN LATERAL (
 SELECT finding_id,correlation_algorithm,correlation_value FROM audit_findings
 WHERE audit_id=$1 AND correlation_group_key COLLATE "C"=walk.group_key
 ORDER BY finding_id COLLATE "C" LIMIT 1) AS l ON true
 LEFT JOIN LATERAL (
 SELECT finding_id,correlation_algorithm,correlation_value FROM audit_findings
 WHERE audit_id=$2 AND correlation_group_key COLLATE "C"=walk.group_key
 ORDER BY finding_id COLLATE "C" LIMIT 1) AS r ON true
 WHERE walk.step>0 ORDER BY walk.step`

// comparisonUnmatchedSQL keeps every unqualified observation as its own row.
// Each side independently contributes at most limit+1 indexed rows.
const comparisonUnmatchedSQL = `WITH selected AS MATERIALIZED (
 (SELECT finding_id,audit_id,correlation_reason FROM audit_findings WHERE audit_id=$1
 AND correlation_group_key COLLATE "C"='' AND finding_id COLLATE "C">$3
 ORDER BY finding_id COLLATE "C" LIMIT $4)
 UNION ALL
 (SELECT finding_id,audit_id,correlation_reason FROM audit_findings WHERE audit_id=$2
 AND correlation_group_key COLLATE "C"='' AND finding_id COLLATE "C">$3
 ORDER BY finding_id COLLATE "C" LIMIT $4)
 ) SELECT ''::text,''::text,''::text,CASE WHEN audit_id=$1 THEN finding_id END,
 CASE WHEN audit_id=$2 THEN finding_id END,correlation_reason
 FROM selected ORDER BY finding_id COLLATE "C" LIMIT $4`

// Member queries seek in Project scope, or union at most two already bounded
// Audit probes. They never scan the Project-wide group to filter a selected pair.
const projectMembersSQL = `SELECT finding_id,audit_id FROM audit_findings WHERE correlation_project_id=$1
 AND correlation_group_key=$2 AND finding_id COLLATE "C">$3
 ORDER BY finding_id COLLATE "C" LIMIT $4`
const auditMembersSQL = `SELECT finding_id,audit_id FROM audit_findings WHERE audit_id=$1
 AND correlation_group_key COLLATE "C"=$2 AND finding_id COLLATE "C">$3
 ORDER BY finding_id COLLATE "C" LIMIT $4`
const pairMembersSQL = `WITH selected AS MATERIALIZED ((` + auditMembersSQL + `)
 UNION ALL (SELECT finding_id,audit_id FROM audit_findings WHERE audit_id=$5
 AND correlation_group_key COLLATE "C"=$2 AND finding_id COLLATE "C">$3
 ORDER BY finding_id COLLATE "C" LIMIT $4))
 SELECT finding_id,audit_id FROM selected ORDER BY finding_id COLLATE "C" LIMIT $4`

// comparisonPreviewsSQL hydrates one bounded batch through actual owner and
// Project authority, using the compact immutable receipt preview only.
const comparisonPreviewsSQL = `SELECT finding.finding_id,finding.audit_id,finding.revision,finding.state,
 receipt.comparison_projection->'preview',decision.verdict,decision.severity,assessment.semantic_assessment,finding.current_assessment_id,finding.current_decision_id,
 receipt.receipt_id,receipt.run_id,receipt.source_run_deleted_at IS NOT NULL,receipt.logical_agent_name,
 receipt.workflow_name,receipt.workflow_version,receipt.workflow_closure_digest
 FROM audit_findings AS finding JOIN audits AS audit USING(audit_id)
 JOIN finding_proposal_receipts AS receipt ON receipt.receipt_id=finding.first_receipt_id
 LEFT JOIN audit_review_decisions AS decision ON decision.decision_id=finding.current_decision_id
 LEFT JOIN audit_finding_assessments AS assessment ON assessment.assessment_id=finding.current_assessment_id
 WHERE audit.owner_id=$1 AND audit.project_id=$2 AND finding.finding_id=ANY($3::text[])`

// checkCorrelationFenceSQL rechecks only scalar revisions after hydration; the
// immutable baseline/profile metadata need not be read or hashed a second time.
const checkCorrelationFenceSQL = `SELECT
 EXISTS(SELECT 1 FROM projects WHERE owner_id=$1 AND project_id=$2
 AND ($3::bigint=0 OR finding_correlation_revision=$3))
 AND (SELECT count(*) FROM unnest($4::text[],$5::bigint[]) AS expected(audit_id,revision)
 JOIN audits AS audit ON audit.audit_id=expected.audit_id AND audit.revision=expected.revision
 WHERE audit.owner_id=$1 AND audit.project_id=$2)=cardinality($4::text[])`
