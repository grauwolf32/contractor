package pgstore

// admissionFrameSQL locks only the closed current Round and its owning active
// Audit. All preparation work stays behind the existing dispatch/deadline fence.
const admissionFrameSQL = `
WITH project_gate AS MATERIALIZED (
 SELECT contractor_require_active_audit_project(project_id,owner_id) FROM audits WHERE audit_id=$1
)
SELECT audit.owner_id,audit.project_id,audit.profile_digest,audit.baseline_snapshot,
 round.ordinal,round.admission_checkpoint
FROM audits AS audit JOIN audit_rounds AS round ON round.audit_id=audit.audit_id CROSS JOIN project_gate
WHERE audit.audit_id=$1 AND round.round_id=$2 AND audit.current_round_id=round.round_id
 AND round.state='closed' AND audit.state='active' AND audit.dispatch_state='open'
 AND (audit.deadline_at IS NULL OR audit.deadline_at>clock_timestamp())
FOR UPDATE OF audit,round`

// admissionSourceUpperSQL freezes receipt traversal separately from the hold
// acceptance time. Later imports of old receipts cannot sneak behind the cursor.
const admissionSourceUpperSQL = `
SELECT COALESCE((SELECT receipt_created_at FROM finding_proposal_audit_holds
 WHERE audit_id=$1 AND created_at<=$2 ORDER BY receipt_created_at DESC,receipt_id DESC LIMIT 1),'0001-01-01'::timestamptz),
 COALESCE((SELECT receipt_id FROM finding_proposal_audit_holds
 WHERE audit_id=$1 AND created_at<=$2 ORDER BY receipt_created_at DESC,receipt_id DESC LIMIT 1),'')`

// admissionCandidateHeadersSQL limits raw held-receipt headers before any
// consumed/rejection/readiness filtering. The existing hold index supplies its
// keyset order; only a bounded prefix of ordinal headers is decoded afterwards.
const admissionCandidateHeadersSQL = `
SELECT hold.receipt_id,hold.receipt_created_at,hold.created_at>$7,
 hold.proposal_ref,
 receipt.checks_projection_bytes,receipt.checks_projection_count
FROM finding_proposal_audit_holds AS hold
CROSS JOIN LATERAL (SELECT checks_projection_bytes,checks_projection_count
 FROM finding_proposal_receipts WHERE receipt_id=hold.receipt_id OFFSET 0) AS receipt
WHERE hold.audit_id=$1 AND hold.project_id=$2
 AND (hold.receipt_created_at,hold.receipt_id)>=($3::timestamptz,$4::text)
 AND ((hold.receipt_created_at,hold.receipt_id)>($3::timestamptz,$4::text)
 OR $9<COALESCE(receipt.checks_projection_count,513)-1)
 AND (hold.receipt_created_at,hold.receipt_id)<=($5::timestamptz,$6::text)
ORDER BY hold.receipt_created_at,hold.receipt_id LIMIT $8`

// One bounded input relation replaces per-candidate status queries. Each EXISTS
// uses the source consume-once key or the natural decision provenance index.
const admissionSourceStatusesSQL = `
SELECT input.position,
 EXISTS(SELECT 1 FROM audit_items AS item JOIN audit_rounds AS round USING(audit_id,round_id)
 WHERE item.audit_id=$1 AND item.proposal_receipt_id=input.receipt_id AND item.proposal_check_ordinal=input.ordinal AND round.state<>'proposed'),
 EXISTS(SELECT 1 FROM audit_artifact_links WHERE audit_id=$1
 AND source_provenance->>'schema'='contractor.audit.admission-decision.v1'
 AND source_provenance->>'source_key'=input.source_key AND source_provenance->>'policy_digest'=$3
 AND source_provenance->>'disposition'='rejected')
FROM jsonb_to_recordset($2::jsonb) AS input(position integer,receipt_id text,ordinal integer,source_key text)`

const admissionFindingSourceSQL = `
SELECT receipt.receipt_id,hold.project_id,receipt.proposal_ref,receipt.proposal_digest,
 receipt.proposal_media_type,receipt.proposal_size_bytes,hold.proposal_ref,receipt.run_id,
 COALESCE(receipt.audit_execution_id,''),COALESCE(receipt.audit_id,''),receipt.checks_projection_bytes,
 CASE WHEN jsonb_array_length(receipt.evidence)<=31 AND jsonb_array_length(hold.evidence)<=31
  AND octet_length(receipt.evidence::text)+octet_length(hold.evidence::text)<=524288 THEN receipt.evidence ELSE '[]'::jsonb END,
 CASE WHEN jsonb_array_length(receipt.evidence)<=31 AND jsonb_array_length(hold.evidence)<=31
  AND octet_length(receipt.evidence::text)+octet_length(hold.evidence::text)<=524288 THEN hold.evidence ELSE '[]'::jsonb END,
 jsonb_array_length(receipt.evidence)>31 OR jsonb_array_length(hold.evidence)>31
  OR octet_length(receipt.evidence::text)+octet_length(hold.evidence::text)>524288
FROM finding_proposal_receipts AS receipt JOIN finding_proposal_audit_holds AS hold USING(receipt_id)
WHERE hold.audit_id=$1 AND receipt.owner_id=$2 AND hold.project_id=$3 AND receipt.receipt_id=$4`

const admissionLinkReplaySQL = `
SELECT EXISTS(SELECT 1 FROM audit_artifact_links WHERE audit_id=$1 AND logical_key=$2),
 EXISTS(SELECT 1 FROM audit_artifact_links WHERE audit_id=$1 AND logical_key=$2
 AND artifact_ref=$3::jsonb AND artifact_digest=$4 AND source_provenance=$5::jsonb)`

const admissionExactChargedSQL = `
SELECT EXISTS(SELECT 1 FROM audit_artifact_links WHERE audit_id=$1
 AND artifact_ref->>'namespace'=$2 AND artifact_ref->>'name'=$3
 AND artifact_ref->>'revision'=$4 AND artifact_digest=$5)`

const admissionCreateProposedSQL = `
INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count,prepared_generation)
SELECT $2,$1,$3,$4::jsonb,$5,'proposed',$6,$7
FROM audits WHERE audit_id=$1 AND state='active' AND dispatch_state='open'
 AND $3<=max_rounds AND $6<=max_items_per_round
 AND (SELECT COALESCE(sum(expected_item_count),0) FROM audit_rounds WHERE audit_id=$1 AND state<>'proposed')+$6<=max_items_total
ON CONFLICT(round_id) DO UPDATE SET prepared_generation=EXCLUDED.prepared_generation
WHERE audit_rounds.audit_id=EXCLUDED.audit_id AND audit_rounds.state='proposed'
 AND audit_rounds.prepared_generation=EXCLUDED.prepared_generation
 AND audit_rounds.expected_item_count=EXCLUDED.expected_item_count
 AND audit_rounds.manifest_ref=EXCLUDED.manifest_ref AND audit_rounds.manifest_digest=EXCLUDED.manifest_digest`

const admissionStageItemsSQL = `
WITH input AS MATERIALIZED (
 SELECT * FROM jsonb_to_recordset($3::jsonb) AS item(item_id text,item_key text,ordinal integer,kind text,subject_key text,
  task_ref jsonb,task_digest text,origin jsonb,workflow_role text,initial_state text,approval_kind text,approval_digest text,
  status text,requested jsonb,completed jsonb,gaps jsonb,rationale text,receipt_id text,proposed_check_ordinal integer,
  proposal_ref jsonb,proposal_digest text,retained_dependencies text,prepared_execution text,
  prepared_page_ordinal integer,prepared_related_count integer)
)
INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role,state,
 approval_kind,approval_subject_digest,coverage_status,coverage_requested,coverage_completed,coverage_gaps,coverage_rationale,
 proposal_receipt_id,proposal_check_ordinal,proposal_ref,proposal_digest,retained_dependencies,prepared_execution,
 prepared_page_ordinal,prepared_related_count)
SELECT item.item_id,$1,$2,item.item_key,item.ordinal,item.kind,item.subject_key,item.task_ref,item.task_digest,item.origin,
 item.workflow_role,item.initial_state,item.approval_kind,NULLIF(item.approval_digest,''),item.status,item.requested,item.completed,
 item.gaps,item.rationale,item.receipt_id,item.proposed_check_ordinal,item.proposal_ref,item.proposal_digest,
 NULLIF(item.retained_dependencies,'')::json,item.prepared_execution::json,item.prepared_page_ordinal,item.prepared_related_count
FROM input AS item JOIN audit_rounds AS round ON round.audit_id=$1 AND round.round_id=$2 AND round.state='proposed'`

const admissionStageReviewsSQL = `
INSERT INTO audit_review_requests(request_id,audit_id,finding_id,subject_kind,subject_id,kind,subject_revision,subject_digest,
 requested_actions,state,expires_at,idempotency_key,request_digest)
SELECT 'review-'||item.item_id,item.audit_id,NULL,'audit-item-action',item.item_id,item.approval_kind,1,item.approval_subject_digest,
 CASE WHEN item.approval_kind='requirement-applicability' THEN '["approve","reject","not_applicable"]'::jsonb
 ELSE '["approve","reject"]'::jsonb END,'pending',audit.deadline_at,'auto:'||item.item_id,item.approval_subject_digest
FROM audit_items AS item JOIN audits AS audit USING(audit_id)
WHERE item.audit_id=$1 AND item.round_id=$2 AND item.ordinal>=$3 AND item.ordinal<$4 AND item.approval_kind<>'none'`

// The descriptor and exact input closure are charged before their JSON values
// are returned to the candidate adapter. Preparation output arrays are bounded
// by the existing role mapping contract.
const admissionPreparationCostSQL = `
SELECT octet_length(output::text)+octet_length((execution.preparation_snapshot->'inputs')::text)+512,
 2+(SELECT count(*)::integer FROM jsonb_object_keys(execution.preparation_snapshot->'inputs'))
FROM audit_executions AS execution CROSS JOIN LATERAL jsonb_array_elements(execution.preparation_outputs) AS output
WHERE execution.audit_id=$1 AND execution.workflow_role=$2 AND execution.role='prepare'
 AND execution.collection_disposition='accepted-result' AND output->>'logicalName'=$3`

// Cost-only preflight touches stored lengths/counts but returns no evidence
// JSON. The same closed CASE as the hydration query caps unsupported closures.
const admissionFindingSourceCostSQL = `
SELECT octet_length(receipt.proposal_ref::text)+octet_length(hold.proposal_ref::text)+2048+
 CASE WHEN jsonb_array_length(receipt.evidence)<=31 AND jsonb_array_length(hold.evidence)<=31
 AND octet_length(receipt.evidence::text)+octet_length(hold.evidence::text)<=524288
 THEN octet_length(receipt.evidence::text)+octet_length(hold.evidence::text) ELSE 4 END,
 2+CASE WHEN jsonb_array_length(receipt.evidence)<=31 AND jsonb_array_length(hold.evidence)<=31
 AND octet_length(receipt.evidence::text)+octet_length(hold.evidence::text)<=524288
 THEN jsonb_array_length(receipt.evidence)+jsonb_array_length(hold.evidence) ELSE 0 END,
 receipt.checks_projection_bytes
FROM finding_proposal_receipts AS receipt JOIN finding_proposal_audit_holds AS hold USING(receipt_id)
WHERE hold.audit_id=$1 AND receipt.owner_id=$2 AND hold.project_id=$3 AND receipt.receipt_id=$4`

// The final bounded staging transaction installs the database membership seal.
// Root publication precedes staging and cannot set these nullable seal fields.
const admissionSealMembershipSQL = `
UPDATE audit_rounds SET acceptance_digest=$4,prepared_page_count=$5,prepared_pages_digest=$6
WHERE audit_id=$1 AND round_id=$2 AND state='proposed'
 AND prepared_generation=$3 AND manifest_digest=$4
 AND prepared_page_count IS NULL AND prepared_pages_digest IS NULL
 AND prepared_staged_count=$7 AND expected_item_count=$7`
