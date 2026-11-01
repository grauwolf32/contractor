-- Source Runs are already gone; exact Audit holdings are the only copies.
INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES('migration-project','migration-owner','project','History','create','sha256:'||repeat('a',64));
INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,
 max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
VALUES('migration-audit','migration-owner','migration-project','profile','1','sha256:'||repeat('a',64),'{}','{}',1,1,10,10,10,1,1048576);
INSERT INTO finding_correlation_fences(project_id) VALUES('migration-project');
INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('project','migration-project');
INSERT INTO artifact_blobs(sha256,payload,size_bytes)
 SELECT sha256(body::bytea),body::bytea,octet_length(body) FROM unnest(ARRAY[
 '{"schema":"contractor.audit.finding-proposal.v1","client_key":"old","title":"Old problem","description":"Original observation","subject":null,"preconditions":[],"standard_refs":[],"evidence_ids":["evidence-1"],"proposed_checks":[],"severity_suggestion":"high","limitations":[]}',
 'original exact evidence','original supported result','original contract','frozen report snapshot']) AS body;
INSERT INTO artifact_versions(version_id,blob_sha256,media_type)
 SELECT 'migration-version-'||row_number() OVER(ORDER BY size_bytes,sha256),sha256,'application/json' FROM artifact_blobs;
INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes)
 SELECT 'project','migration-project','audit-migration-audit',version_id,'r1',true,1,blob.size_bytes
 FROM artifact_versions AS version JOIN artifact_blobs AS blob ON blob.sha256=version.blob_sha256;
INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
 SELECT scope_kind,scope_id,namespace,name,current_revision,name FROM artifact_bindings;
INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,
 logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,
 workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,
 proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection,retention_state,source_run_deleted_at)
 SELECT 'receipt-'||suffix,'proposal-'||suffix,'allocation-'||suffix,'runtime','instance','stage','worker','invocation-'||suffix,
 'submission-'||suffix,suffix,'sha256:'||repeat('a',64),'gone-run-'||suffix,'migration-owner','migration-project',
 'source','1','contractor/v1alpha1','{"name":"source","version":"1"}','sha256:'||repeat('b',64),
 jsonb_build_object('namespace','source','name','old','revision','r1'),'sha256:'||encode(blob.sha256,'hex'),'application/json',blob.size_bytes,
 '[]','{"title":"Old problem","description":"Original observation","subject":null,"preconditions":[],"standard_refs":[],"severity_suggestion":"high","limitations":[]}',
 'audit-held',clock_timestamp()
 FROM artifact_versions AS version JOIN artifact_blobs AS blob ON blob.sha256=version.blob_sha256
 CROSS JOIN unnest(ARRAY['confirmed','duplicate']) AS suffix WHERE convert_from(blob.payload,'UTF8') LIKE '{"schema"%';
INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
 SELECT receipt.receipt_id,'migration-audit','migration-project',
 jsonb_build_object('ref',jsonb_build_object('namespace','audit-migration-audit','name',version.version_id,'revision','r1'),
 'digest',receipt.proposal_digest,'mediaType','application/json','sizeBytes',receipt.proposal_size_bytes),
 jsonb_build_array((SELECT jsonb_build_object('ref',jsonb_build_object('namespace','audit-migration-audit','name',v.version_id,'revision','r1'),
 'digest','sha256:'||encode(b.sha256,'hex'),'mediaType','application/json','sizeBytes',b.size_bytes)
 FROM artifact_versions AS v JOIN artifact_blobs AS b ON b.sha256=v.blob_sha256 WHERE b.payload='original exact evidence'::bytea))
 FROM finding_proposal_receipts AS receipt JOIN artifact_blobs AS blob ON 'sha256:'||encode(blob.sha256,'hex')=receipt.proposal_digest
 JOIN artifact_versions AS version ON version.blob_sha256=blob.sha256;
INSERT INTO audit_finding_assessments(assessment_id,finding_id,audit_id,receipt_id,semantic_assessment,
 result_ref,result_digest,direct_verification,contract_ref,contract_digest)
 SELECT 'old-supported',finding_id,audit_id,first_receipt_id,'supported',
 (SELECT jsonb_build_object('namespace','audit-migration-audit','name',v.version_id,'revision','r1') FROM artifact_versions v JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE b.payload='original supported result'::bytea),
 'sha256:'||encode(sha256('original supported result'::bytea),'hex'),true,
 (SELECT jsonb_build_object('namespace','audit-migration-audit','name',v.version_id,'revision','r1') FROM artifact_versions v JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE b.payload='original contract'::bytea),
 'sha256:'||encode(sha256('original contract'::bytea),'hex') FROM audit_findings WHERE first_receipt_id='receipt-confirmed';
INSERT INTO audit_review_requests(request_id,audit_id,finding_id,kind,subject_revision,subject_digest,requested_actions,state,
 idempotency_key,request_digest,subject_kind,subject_id)
 SELECT 'review-'||first_receipt_id,audit_id,finding_id,'finding-triage',1,'sha256:'||repeat('a',64),'[]','decided',
 'review-'||first_receipt_id,'sha256:'||repeat('a',64),'finding',finding_id FROM audit_findings;
INSERT INTO audit_review_decisions(decision_id,request_id,audit_id,finding_id,actor_id,action,verdict,severity,rationale,
 duplicate_target_id,subject_revision,subject_digest,idempotency_key,request_digest)
 SELECT 'decision-'||f.first_receipt_id,'review-'||f.first_receipt_id,f.audit_id,f.finding_id,'migration-owner',
 CASE WHEN f.first_receipt_id='receipt-confirmed' THEN 'true_positive' ELSE 'duplicate' END,
 CASE WHEN f.first_receipt_id='receipt-confirmed' THEN 'true_positive' ELSE 'duplicate' END,
 CASE WHEN f.first_receipt_id='receipt-confirmed' THEN 'high' END,'Original analyst annotation',
 CASE WHEN f.first_receipt_id='receipt-duplicate' THEN (SELECT finding_id FROM audit_findings WHERE first_receipt_id='receipt-confirmed') END,
 1,'sha256:'||repeat('a',64),'decision-'||f.first_receipt_id,'sha256:'||repeat('a',64) FROM audit_findings AS f;
UPDATE audit_findings AS f SET state=CASE WHEN first_receipt_id='receipt-confirmed' THEN 'confirmed' ELSE 'duplicate' END,
 current_assessment_id=CASE WHEN first_receipt_id='receipt-confirmed' THEN 'old-supported' END,
 current_decision_id=CASE WHEN first_receipt_id='receipt-confirmed' THEN 'decision-'||first_receipt_id END,
 duplicate_target_id=CASE WHEN first_receipt_id='receipt-duplicate' THEN (SELECT finding_id FROM audit_findings WHERE first_receipt_id='receipt-confirmed') END;
