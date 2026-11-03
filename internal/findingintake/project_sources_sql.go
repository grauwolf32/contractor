package findingintake

const getProjectReceiptSQL = `SELECT ` + receiptProjection + `
FROM finding_proposal_receipts AS receipt JOIN projects AS project ON project.project_id=receipt.project_id
WHERE project.owner_id=$1 AND project.kind='project' AND project.lifecycle_state='active' AND receipt.owner_id=$1
AND receipt.project_id=$2 AND receipt.receipt_id=$3 AND receipt.project_proposal IS NOT NULL`

const authorizeProjectSourcesSQL = `SELECT fence.revision FROM projects AS project
JOIN finding_correlation_fences AS fence USING(project_id)
WHERE project.owner_id=$1 AND project.project_id=$2 AND project.kind='project' AND project.lifecycle_state='active'
AND ($3::text='' OR EXISTS(SELECT 1 FROM project_findings WHERE project_id=$2 AND finding_id=$3))`

const authorizeProjectSourceHistorySQL = `SELECT EXISTS(
 SELECT 1 FROM finding_proposal_receipts WHERE owner_id=$1 AND project_id=$2 AND receipt_id=$3 AND project_proposal IS NOT NULL)`

const projectSourcePreviewColumns = `receipt.receipt_id,receipt.proposal_id,receipt.project_id,source.finding_id,
receipt.project_proposal,jsonb_array_length(receipt.project_evidence),receipt.publication_requested,receipt.created_at,
receipt.run_id,receipt.stage_execution_id,receipt.allocation_id,receipt.invocation_id,receipt.logical_agent_name,
receipt.workflow_name,receipt.workflow_version,receipt.workflow_schema_version,receipt.workflow_configuration_ref,receipt.workflow_closure_digest,
receipt.audit_id,receipt.audit_execution_id,receipt.audit_task_id,receipt.audit_hypothesis,receipt.source_run_deleted_at`

// listProjectSourcesSQL seeks retained historical materials without reading
// proposal/evidence payloads or walking another Project's receipts.
const listProjectSourcesSQL = `SELECT ` + projectSourcePreviewColumns + `
FROM finding_proposal_receipts AS receipt
LEFT JOIN project_finding_sources AS source USING(receipt_id)
WHERE receipt.owner_id=$1 AND receipt.project_id=$2 AND $3::text=''
AND receipt.project_proposal IS NOT NULL
AND (receipt.created_at,receipt.receipt_id COLLATE "C")>(coalesce($4::timestamptz,'-infinity'),$5::text COLLATE "C")
ORDER BY receipt.created_at,receipt.receipt_id COLLATE "C" LIMIT $6`

// listCanonicalFindingSourcesSQL uses this Finding's source index, rather
// than filtering a whole Project history under an optional OR predicate.
const listCanonicalFindingSourcesSQL = `SELECT ` + projectSourcePreviewColumns + `
FROM project_finding_sources AS source
JOIN finding_proposal_receipts AS receipt USING(receipt_id)
WHERE source.finding_id=$3 AND source.project_id=$2 AND receipt.owner_id=$1
AND receipt.project_proposal IS NOT NULL
AND (source.created_at,source.receipt_id COLLATE "C")>(coalesce($4::timestamptz,'-infinity'),$5::text COLLATE "C")
ORDER BY source.created_at,source.receipt_id COLLATE "C" LIMIT $6`
