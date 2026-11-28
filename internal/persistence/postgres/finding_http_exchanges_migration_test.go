package postgres

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type httpMigrationFixture struct {
	ctx                    context.Context
	pool                   *pgxpool.Pool
	old, current           [][]byte
	descriptors            []map[string]any
	historyOld, historyNew []byte
	evidence               []byte
}

func newHTTPMigrationFixture(t *testing.T, count int, receipts bool) httpMigrationFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	t.Cleanup(cancel)
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	if len(available) < 3 || available[2].version != 3 {
		t.Fatal("missing HTTP format migration")
	}
	if _, err := applyMigrations(ctx, pool, MigrationBudgets{}, available[:2]); err != nil {
		t.Fatal(err)
	}
	f := httpMigrationFixture{ctx: ctx, pool: pool, evidence: []byte(`{"schema":"test.http.evidence","data":"unchanged"}`)}
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('project_http_migration','owner','project','HTTP migration','create','sha256:'||repeat('a',64));
 INSERT INTO artifact_scopes(scope_kind,scope_id) VALUES('project','project_http_migration')`); err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		evidence, err := seedHTTPMigrationArtifact(ctx, tx, "evidence", f.evidence, "application/vnd.contractor.http-exchange+json")
		if err != nil {
			return err
		}
		for i := range count {
			status := 200
			proposal := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: fmt.Sprintf("client-%d", i), Title: "Проверка <tag> 😀", Description: "Line\nquote \" and \u2028", Subject: &auditdomain.FindingSubject{Kind: "endpoint", Key: "target"}, Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{"e1"}, ProposedChecks: []auditdomain.ProposedCheck{{Objective: "Inspect the selected evidence", Method: "manual"}}, SeveritySuggestion: "medium", Limitations: []string{"A bounded fixture"}, HTTPExchanges: []auditdomain.FindingHTTPExchange{{RequestID: 9007199254740991, RequestTag: "tag", Attempts: []auditdomain.FindingHTTPAttempt{{Method: "GET", URL: "https://target.test/", Headers: []auditdomain.FindingHTTPHeader{}, Status: &status}}, ResponseBodyEvidenceID: "e1"}}}
			current, err := auditdomain.EncodeFindingProposal(proposal)
			if err != nil {
				return err
			}
			var oldDoc map[string]any
			if err := json.Unmarshal(current, &oldDoc); err != nil {
				return err
			}
			oldDoc["http_exchange"] = oldDoc["http_exchanges"].([]any)[0]
			delete(oldDoc, "http_exchanges")
			old, err := strictjson.Canonical(oldDoc)
			if err != nil {
				return err
			}
			descriptor, err := seedHTTPMigrationArtifact(ctx, tx, fmt.Sprintf("proposal-%d", i), old, "application/json")
			if err != nil {
				return err
			}
			f.old = append(f.old, old)
			f.current = append(f.current, current)
			f.descriptors = append(f.descriptors, descriptor)
			if !receipts {
				continue
			}
			projection := auditdomain.ProjectFindingReport(proposal)
			report, err := auditdomain.EncodeFindingReportProjection(projection)
			if err != nil {
				return err
			}
			var reportDoc map[string]any
			_ = json.Unmarshal(report, &reportDoc)
			reportDoc["http_exchange"] = reportDoc["http_exchanges"].([]any)[0]
			delete(reportDoc, "http_exchanges")
			oldReport, err := strictjson.Canonical(reportDoc)
			if err != nil {
				return err
			}
			checks, err := auditdomain.BuildFindingChecksProjection(current, proposal)
			if err != nil {
				return err
			}
			checks.ProposalDigest = contentdigest.Bytes(old)
			checks.SizeBytes = int64(len(old))
			checks = legacyHTTPFragments(t, old, checks)
			checksJSON, err := auditdomain.EncodeFindingChecksProjection(checks)
			if err != nil {
				return err
			}
			exactJSON, _ := json.Marshal(descriptor)
			refJSON, _ := json.Marshal(descriptor["ref"])
			evidenceJSON, _ := json.Marshal([]any{evidence})
			_, err = tx.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence,report_projection,checks_projection,retention_state,project_proposal,project_evidence)
 VALUES($1,$2,'allocation','agent','instance','stage','worker','invocation',$3,$4,$5,'run','owner','project_http_migration','workflow','1','contractor/v1alpha1','{}',$5,$6,$7,'application/json',$8,$9,$10,$11,'project-held',$12,$9)`, fmt.Sprintf("receipt_%d", i), fmt.Sprintf("proposal_%d", i), fmt.Sprintf("submission-%d", i), proposal.ClientKey, "sha256:"+strings.Repeat("a", 64), refJSON, contentdigest.Bytes(old), len(old), evidenceJSON, string(oldReport), checksJSON, exactJSON)
			if err != nil {
				return err
			}
			_, err = tx.Exec(ctx, `INSERT INTO project_findings(owner_id,finding_id,project_id,first_receipt_id,first_proposal_ref,title,subject_kind,subject_key,subject_truncated,severity_suggestion,correlation_algorithm,correlation_value,correlation_group_key,correlation_reason,state,analyst_verdict,analyst_severity,rationale,actor_id,assessed_at,revision)
 VALUES('owner',$1,'project_http_migration',$2,$3,$4,'endpoint','target',false,'medium','','','','not_recorded','confirmed','true_positive','medium','Reviewed','owner','2026-11-01T00:00:00Z',7)`, fmt.Sprintf("finding_%d", i), fmt.Sprintf("receipt_%d", i), exactJSON, proposal.Title)
			if err != nil {
				return err
			}
			_, err = tx.Exec(ctx, `INSERT INTO project_finding_sources(receipt_id,project_id,finding_id) VALUES($1,'project_http_migration',$2)`, fmt.Sprintf("receipt_%d", i), fmt.Sprintf("finding_%d", i))
			if err != nil {
				return err
			}
		}
		if _, err = seedHTTPMigrationArtifact(ctx, tx, "current-format", bytes.ReplaceAll(f.current[0], []byte("client-0"), []byte("client-control")), "application/json"); err != nil {
			return err
		}
		if receipts {
			history := map[string]any{"schema": auditdomain.FindingSourceHistorySchema, "kind": "audit-import", "receiptId": "receipt_0", "original": map[string]any{"runId": "run"}, "materials": map[string]any{"proposal": f.descriptors[0], "evidence": []any{evidence}}}
			f.historyOld, err = strictjson.Canonical(history)
			if err != nil {
				return err
			}
			if _, err = seedHTTPMigrationArtifact(ctx, tx, "history", f.historyOld, "application/json"); err != nil {
				return err
			}
			expected := cloneHTTPDescriptor(f.descriptors[0], f.current[0])
			history["materials"].(map[string]any)["proposal"] = expected
			f.historyNew, err = strictjson.Canonical(history)
			if err != nil {
				return err
			}
			if _, err = tx.Exec(ctx, `INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
 VALUES('audit_http_migration','owner','project_http_migration','profile','1','sha256:'||repeat('a',64),'{}','{}',1,1,10,10,10,1,1024)`); err != nil {
				return err
			}
			exactJSON, _ := json.Marshal(f.descriptors[0])
			evidenceJSON, _ := json.Marshal([]any{evidence})
			if _, err = tx.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence) VALUES('receipt_0','audit_http_migration','project_http_migration',$1,$2)`, exactJSON, evidenceJSON); err != nil {
				return err
			}
			response, _ := strictjson.Canonical(map[string]any{"finding": map[string]any{"firstProposal": f.descriptors[0]}})
			_, _, err = KeyedMutation(ctx, tx, ProjectReceiptKey("owner", "project_http_migration", "finding_0", "finding.assess", "key"), "sha256:"+strings.Repeat("b", 64), func() (Receipt, error) {
				return Receipt{Operation: "finding.assess", ResourceID: "finding_0", OwnerID: "owner", ProjectID: "project_http_migration", Response: response}, nil
			})
			if err != nil {
				return err
			}
		}
		_, err = seedHTTPMigrationArtifact(ctx, tx, "unrelated", []byte(`{"unrelated":"http_exchange","data":"<unchanged>","ref":{},"digest":"not-a-digest","sizeBytes":0}`), "application/json")
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func seedHTTPMigrationArtifact(ctx context.Context, tx pgx.Tx, name string, data []byte, media string) (map[string]any, error) {
	digest := contentdigest.Bytes(data)
	_, err := tx.Exec(ctx, `INSERT INTO artifact_blobs(sha256,payload,size_bytes) VALUES(sha256($1::bytea),$1,$2) ON CONFLICT(sha256) DO NOTHING`, data, len(data))
	if err != nil {
		return nil, err
	}
	_, err = tx.Exec(ctx, `INSERT INTO artifact_versions(version_id,blob_sha256,media_type) VALUES($1,sha256($2::bytea),$3)`, "version_"+name, data, media)
	if err != nil {
		return nil, err
	}
	_, err = tx.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes) VALUES('project','project_http_migration','finding-materials',$1,$2,true,1,$3)`, name, "rev_"+name, len(data))
	if err != nil {
		return nil, err
	}
	_, err = tx.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id) VALUES('project','project_http_migration','finding-materials',$1,$2,$3)`, name, "rev_"+name, "version_"+name)
	if err != nil {
		return nil, err
	}
	return map[string]any{"ref": map[string]any{"namespace": "finding-materials", "name": name, "revision": "rev_" + name}, "digest": digest, "mediaType": media, "sizeBytes": len(data)}, nil
}

func legacyHTTPFragments(t *testing.T, raw []byte, projection auditdomain.FindingChecksProjection) auditdomain.FindingChecksProjection {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(raw))
	_, err := decoder.Token()
	if err != nil {
		t.Fatal(err)
	}
	fragment := func(value []byte) auditdomain.FindingFragment {
		return auditdomain.FindingFragment{Offset: decoder.InputOffset() - int64(len(value)), Length: int64(len(value)), Digest: contentdigest.Bytes(value)}
	}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			t.Fatal(err)
		}
		if key == "proposed_checks" {
			_, _ = decoder.Token()
			i := 0
			for decoder.More() {
				var value json.RawMessage
				if err = decoder.Decode(&value); err != nil {
					t.Fatal(err)
				}
				projection.Checks[i].Fragment = fragment(value)
				i++
			}
			_, _ = decoder.Token()
			continue
		}
		var value json.RawMessage
		if err = decoder.Decode(&value); err != nil {
			t.Fatal(err)
		}
		f := fragment(value)
		if key == "subject" {
			projection.Subject = &f
		}
		if key == "limitations" {
			projection.Limitations = &f
		}
	}
	return projection
}
func cloneHTTPDescriptor(old map[string]any, data []byte) map[string]any {
	return map[string]any{"ref": old["ref"], "digest": contentdigest.Bytes(data), "mediaType": old["mediaType"], "sizeBytes": len(data)}
}

func httpMigrationSnapshot(t *testing.T, f httpMigrationFixture) string {
	t.Helper()
	var result string
	err := f.pool.QueryRow(f.ctx, `SELECT jsonb_build_object('blobs',(SELECT jsonb_agg(to_jsonb(b) ORDER BY sha256) FROM artifact_blobs b),'versions',(SELECT jsonb_agg(to_jsonb(v) ORDER BY version_id) FROM artifact_versions v),'bindings',(SELECT jsonb_agg(to_jsonb(b) ORDER BY scope_kind,scope_id,namespace,name) FROM artifact_bindings b),'revisions',(SELECT jsonb_agg(to_jsonb(r) ORDER BY scope_kind,scope_id,namespace,name,revision) FROM artifact_binding_revisions r),'receipts',(SELECT jsonb_agg(to_jsonb(r) ORDER BY receipt_id) FROM finding_proposal_receipts r),'findings',(SELECT jsonb_agg(to_jsonb(f) ORDER BY finding_id) FROM project_findings f),'triggers',(SELECT jsonb_agg(jsonb_build_array(tgrelid,tgname,tgenabled) ORDER BY tgrelid,tgname) FROM pg_trigger WHERE NOT tgisinternal AND tgrelid IN (SELECT oid FROM pg_class WHERE relnamespace=current_schema()::regnamespace)))::text`).Scan(&result)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestFindingHTTPFormatMigrationPreservesIdentitiesWithoutArchives(t *testing.T) {
	f := newHTTPMigrationFixture(t, 2, true)
	var beforeCounts string
	if err := f.pool.QueryRow(f.ctx, `SELECT jsonb_build_array((SELECT count(*) FROM artifact_blobs),(SELECT count(*) FROM artifact_versions),(SELECT count(*) FROM artifact_bindings),(SELECT count(*) FROM artifact_binding_revisions),(SELECT sum(hold_count) FROM artifact_binding_revisions))::text`).Scan(&beforeCounts); err != nil {
		t.Fatal(err)
	}
	var guards string
	if err := f.pool.QueryRow(f.ctx, `SELECT jsonb_agg(jsonb_build_array(tgrelid,tgname,tgenabled) ORDER BY tgrelid,tgname)::text FROM pg_trigger WHERE NOT tgisinternal AND tgrelid IN (SELECT oid FROM pg_class WHERE relnamespace=current_schema()::regnamespace)`).Scan(&guards); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(f.ctx, f.pool, MigrationBudgets{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.AppliedVersions, []int64{3}) {
		t.Fatalf("applied=%v", result.AppliedVersions)
	}
	for i, expected := range f.current {
		var actual, checks []byte
		var digest, report string
		var size int64
		err = f.pool.QueryRow(f.ctx, `SELECT b.payload,r.proposal_digest,r.proposal_size_bytes,r.report_projection,r.checks_projection FROM finding_proposal_receipts r JOIN artifact_versions v ON v.version_id=$1 JOIN artifact_blobs b ON b.sha256=v.blob_sha256 WHERE r.receipt_id=$2`, fmt.Sprintf("version_proposal-%d", i), fmt.Sprintf("receipt_%d", i)).Scan(&actual, &digest, &size, &report, &checks)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(actual, expected) || digest != contentdigest.Bytes(expected) || size != int64(len(expected)) {
			t.Fatal("proposal bytes or exact identity were not reconciled")
		}
		proposal, err := auditdomain.DecodeFindingProposal(actual)
		if err != nil {
			t.Fatal(err)
		}
		expectedReport, _ := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(proposal))
		if string(expectedReport) != report {
			t.Fatal("report is not the current canonical projection")
		}
		expectedChecks, err := auditdomain.BuildFindingChecksProjection(actual, proposal)
		if err != nil {
			t.Fatal(err)
		}
		expectedCheckBytes, _ := auditdomain.EncodeFindingChecksProjection(expectedChecks)
		canonicalChecks, err := strictjson.Canonical(json.RawMessage(checks))
		if err != nil || !bytes.Equal(canonicalChecks, expectedCheckBytes) {
			t.Fatalf("stale fragment offsets/digests: %v", err)
		}
		var exact map[string]any
		var raw []byte
		var revision int
		var rationale string
		if err := f.pool.QueryRow(f.ctx, `SELECT first_proposal_ref,revision,rationale FROM project_findings WHERE finding_id=$1`, fmt.Sprintf("finding_%d", i)).Scan(&raw, &revision, &rationale); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &exact); err != nil {
			t.Fatal(err)
		}
		if exact["digest"] != digest || revision != 7 || rationale != "Reviewed" {
			t.Fatal("Finding descriptor or analyst identity changed")
		}
	}
	var history, evidence []byte
	if err := f.pool.QueryRow(f.ctx, `SELECT payload FROM artifact_blobs b JOIN artifact_versions v ON v.blob_sha256=b.sha256 WHERE v.version_id='version_history'`).Scan(&history); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(history, f.historyNew) {
		t.Fatal("source history exact references not updated canonically")
	}
	if err := f.pool.QueryRow(f.ctx, `SELECT payload FROM artifact_blobs b JOIN artifact_versions v ON v.blob_sha256=b.sha256 WHERE v.version_id='version_evidence'`).Scan(&evidence); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(evidence, f.evidence) {
		t.Fatal("unchanged HTTP evidence changed")
	}
	var hold, response []byte
	if err := f.pool.QueryRow(f.ctx, `SELECT proposal_ref FROM finding_proposal_audit_holds WHERE receipt_id='receipt_0'`).Scan(&hold); err != nil {
		t.Fatal(err)
	}
	expectedDescriptor, _ := strictjson.Canonical(cloneHTTPDescriptor(f.descriptors[0], f.current[0]))
	canonicalHold, err := strictjson.Canonical(json.RawMessage(hold))
	if err != nil || !bytes.Equal(canonicalHold, expectedDescriptor) {
		t.Fatal("audit hold exact reference was not reconciled")
	}
	if err := f.pool.QueryRow(f.ctx, `SELECT response FROM mutation_receipts`).Scan(&response); err != nil {
		t.Fatal(err)
	}
	expectedResponse, _ := strictjson.Canonical(map[string]any{"finding": map[string]any{"firstProposal": cloneHTTPDescriptor(f.descriptors[0], f.current[0])}})
	if !bytes.Equal(response, expectedResponse) {
		t.Fatal("mutation receipt exact reference was not reconciled")
	}
	var afterCounts, afterGuards string
	if err := f.pool.QueryRow(f.ctx, `SELECT jsonb_build_array((SELECT count(*) FROM artifact_blobs),(SELECT count(*) FROM artifact_versions),(SELECT count(*) FROM artifact_bindings),(SELECT count(*) FROM artifact_binding_revisions),(SELECT sum(hold_count) FROM artifact_binding_revisions))::text`).Scan(&afterCounts); err != nil {
		t.Fatal(err)
	}
	if beforeCounts != afterCounts {
		t.Fatalf("published duplicate/archive content: %s -> %s", beforeCounts, afterCounts)
	}
	if err := f.pool.QueryRow(f.ctx, `SELECT jsonb_agg(jsonb_build_array(tgrelid,tgname,tgenabled) ORDER BY tgrelid,tgname)::text FROM pg_trigger WHERE NOT tgisinternal AND tgrelid IN (SELECT oid FROM pg_class WHERE relnamespace=current_schema()::regnamespace)`).Scan(&afterGuards); err != nil {
		t.Fatal(err)
	}
	if guards != afterGuards {
		t.Fatal("immutable trigger modes were not restored")
	}
	var invalid int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_bindings b WHERE retained_bytes<>(SELECT sum(bl.size_bytes) FROM artifact_binding_revisions r JOIN artifact_versions v USING(version_id) JOIN artifact_blobs bl ON bl.sha256=v.blob_sha256 WHERE (r.scope_kind,r.scope_id,r.namespace,r.name)=(b.scope_kind,b.scope_id,b.namespace,b.name))`).Scan(&invalid); err != nil || invalid != 0 {
		t.Fatalf("byte counters invalid: %d %v", invalid, err)
	}
	before := httpMigrationSnapshot(t, f)
	if result, err := ApplyMigrations(f.ctx, f.pool, MigrationBudgets{}); err != nil || len(result.AppliedVersions) != 0 {
		t.Fatalf("rerun=%+v %v", result, err)
	}
	if httpMigrationSnapshot(t, f) != before {
		t.Fatal("rerun changed content")
	}
	_, err = f.pool.Exec(f.ctx, `UPDATE artifact_versions SET media_type='text/plain' WHERE version_id='version_history'`)
	assertSQLState(t, err, SQLStateCheckViolation)
}

func TestFindingHTTPFormatMigrationRollsBackMalformedOrUnsupportedData(t *testing.T) {
	for _, kind := range []string{"mixed-format", "unknown-reference", "late-descriptor-error", "noncanonical-proposal", "noninteger-history", "oversized-proposal", "collection-package"} {
		t.Run(kind, func(t *testing.T) {
			f := newHTTPMigrationFixture(t, 1, true)
			if kind == "late-descriptor-error" {
				_, err := f.pool.Exec(f.ctx, `ALTER TABLE project_findings DISABLE TRIGGER project_findings_protect_identity; UPDATE project_findings SET first_proposal_ref=jsonb_set(first_proposal_ref,'{sizeBytes}',to_jsonb((first_proposal_ref->>'sizeBytes')::bigint+1)); ALTER TABLE project_findings ENABLE TRIGGER project_findings_protect_identity`)
				if err != nil {
					t.Fatal(err)
				}
			} else if kind == "collection-package" {
				if err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
					_, err := seedHTTPMigrationArtifact(f.ctx, tx, "collection", []byte("opaque package"), "application/vnd.contractor.findings-collection+zip")
					return err
				}); err != nil {
					t.Fatal(err)
				}
			} else {
				var data []byte
				if kind == "noncanonical-proposal" {
					data = append([]byte(" "), f.old[0]...)
				} else if kind == "noninteger-history" {
					data, _ = strictjson.Canonical(map[string]any{"schema": auditdomain.FindingSourceHistorySchema, "artifact": f.descriptors[0], "fraction": 0.5})
				} else if kind == "oversized-proposal" {
					var v map[string]any
					_ = json.Unmarshal(f.old[0], &v)
					checks := make([]any, 512)
					for i := range checks {
						checks[i] = map[string]any{"objective": "x", "method": "manual"}
					}
					v["proposed_checks"] = checks
					small, _ := strictjson.Canonical(v)
					extra := auditdomain.MaximumDocumentBytes - len(small)
					for i := range checks {
						count := extra / 512
						if i < extra%512 {
							count++
						}
						checks[i].(map[string]any)["objective"] = strings.Repeat("x", 1+count)
					}
					data, _ = strictjson.Canonical(v)
					if len(data) != auditdomain.MaximumDocumentBytes {
						t.Fatal("size-bound fixture is not exact")
					}
				} else if kind == "mixed-format" {
					var v map[string]any
					_ = json.Unmarshal(f.old[0], &v)
					v["http_exchanges"] = []any{v["http_exchange"]}
					data, _ = strictjson.Canonical(v)
				} else {
					data, _ = strictjson.Canonical(map[string]any{"schema": "unknown.snapshot.v1", "artifact": f.descriptors[0]})
				}
				if err := InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
					_, err := seedHTTPMigrationArtifact(f.ctx, tx, "unsupported", data, "application/json")
					return err
				}); err != nil {
					t.Fatal(err)
				}
			}
			before := httpMigrationSnapshot(t, f)
			if _, err := ApplyMigrations(f.ctx, f.pool, MigrationBudgets{}); err == nil {
				t.Fatal("unsupported migration succeeded")
			}
			if httpMigrationSnapshot(t, f) != before {
				t.Fatal("failed migration changed bytes, metadata or trigger modes")
			}
			var version int
			if err := f.pool.QueryRow(f.ctx, `SELECT max(version) FROM contractor_schema_migrations`).Scan(&version); err != nil || version != 2 {
				t.Fatalf("failed migration advanced ledger: %d %v", version, err)
			}
		})
	}
}

func TestFindingHTTPFormatMigrationBoundsRewriteCardinality(t *testing.T) {
	for _, count := range []int{512, 513} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			f := newHTTPMigrationFixture(t, count, false)
			before := httpMigrationSnapshot(t, f)
			_, err := ApplyMigrations(f.ctx, f.pool, MigrationBudgets{})
			if count == 512 {
				if err != nil {
					t.Fatal(err)
				}
				var total int
				if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_blobs WHERE payload LIKE $1`, []byte(`%"http_exchanges"%`)).Scan(&total); err != nil || total != 513 {
					t.Fatalf("converted=%d %v", total, err)
				}
			} else if err == nil || httpMigrationSnapshot(t, f) != before {
				t.Fatal("over-bound migration did not roll back")
			}
		})
	}
}
