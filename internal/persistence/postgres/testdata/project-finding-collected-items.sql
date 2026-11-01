-- A collected check whose producer has already been normally retired.
DO $$
DECLARE ref jsonb; digest text;
BEGIN
 SELECT jsonb_build_object('namespace',r.namespace,'name',r.name,'revision',r.revision),
 'sha256:'||encode(b.sha256,'hex') INTO ref,digest
 FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id)
 JOIN artifact_blobs b ON b.sha256=v.blob_sha256 LIMIT 1;
 INSERT INTO audit_rounds(round_id,audit_id,ordinal,manifest_ref,manifest_digest,state,expected_item_count)
 VALUES('collected-round','migration-audit',1,ref,digest,'closed',1);
 INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,workflow_role,state,origin,coverage_status,coverage_requested,coverage_completed,coverage_gaps)
 VALUES('collected-item','migration-audit','collected-round','check',0,'check','component',ref,digest,'check','ready','{"schema":"contractor.audit.item-origin.v1","entryKey":"check","provenanceIncomplete":true}','not-tested','[]','[]','[]');
 INSERT INTO audit_executions(execution_id,audit_id,round_id,role,workflow_role,lane_class,manifest_ref,manifest_digest,submission_key,request_digest,run_id,state,terminal_outcome,terminal_run_generation,terminal_run_sequence,terminal_observed_at,run_provenance,run_deleted_at,collection_receipt_id,collection_disposition,collection_source_output_ref,collection_source_output_digest,collection_retained_refs,collection_request_digest,collected_at)
 VALUES('collected-execution','migration-audit','collected-round','check','check','tool',ref,digest,'collected-submission',digest,'collected-old-run','collected','succeeded','generation',1,clock_timestamp(),'{"schema":"contractor.audit.run-provenance.v1","runId":"collected-old-run"}',clock_timestamp(),'collected-receipt','accepted-result',ref,digest,'[]',digest,clock_timestamp());
 INSERT INTO audit_execution_items(execution_item_id,execution_id,audit_id,round_id,item_id,batch_ordinal,item_attempt,task_ref,task_digest,input_refs,state,collection_disposition,result_ref,result_digest,collected_at)
 VALUES('collected-member','collected-execution','migration-audit','collected-round','collected-item',0,1,ref,digest,'[]','settled','accepted-result',ref,digest,clock_timestamp());
 UPDATE audit_items SET last_execution_item_id='collected-member',state='settled',final_disposition='accepted-result',accepted_result_ref=ref,accepted_result_digest=digest WHERE item_id='collected-item';
END $$;
