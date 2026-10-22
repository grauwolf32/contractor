package postgres

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPostgresFindingCorrelationMigrationFreezesRetainedPrimaryKeys(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx := t.Context()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 23)
	if _, err := pool.Exec(ctx, `INSERT INTO projects(project_id,owner_id,kind,name,request_idempotency_key,request_digest)
 VALUES('correlation-upgrade','owner','project','Retained','create','sha256:'||repeat('a',64));
 INSERT INTO audits(audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,profile_snapshot,input_selection,
 max_rounds,batch_size,max_items_per_round,max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes)
 VALUES('upgrade-audit','owner','correlation-upgrade','profile','1','sha256:'||repeat('a',64),'{}','{}',1,1,10,10,10,1,1024)`); err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name, File, ZipBase64 string
			Line                  int64
			QualifyLocal          bool
			Derived               json.RawMessage
			Keys                  []struct {
				SiteKey  string `json:"siteKey"`
				IssueKey string `json:"issueKey,omitempty"`
			}
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-anchor-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	type selected struct {
		name, algorithm, value, reason string
		anchors                        []byte
	}
	cases := []selected{{name: "old", reason: "not_recorded", anchors: []byte(`{"schema":"contractor.finding.anchors-projection.v1","availability":"unavailable","reason":"not_recorded"}`)}}
	for _, c := range fixture.Cases {
		if c.Name != "python-site" && c.Name != "python-local-key" {
			continue
		}
		var basis struct{ SourceRef contracts.ArtifactRef }
		if err := json.Unmarshal(c.Derived, &basis); err != nil {
			t.Fatal(err)
		}
		data, err := base64.StdEncoding.DecodeString(c.ZipBase64)
		if err != nil {
			t.Fatal(err)
		}
		source := contracts.ExactContent(basis.SourceRef, "application/zip", data)
		anchors, err := json.Marshal(map[string]any{"schema": "contractor.finding.anchors-projection.v1", "availability": "available", "repository": "https://example.test:443/repo.git", "commit": strings.Repeat("a", 40), "source": source, "basis": c.Derived, "keys": c.Keys})
		if err != nil {
			t.Fatal(err)
		}
		item := selected{name: c.Name, anchors: anchors, reason: "no_qualified_key"}
		if c.QualifyLocal {
			item.algorithm = "contractor/snapshot-issue-anchor/v1"
			item.value = c.Keys[0].IssueKey
			item.reason = ""
		}
		cases = append(cases, item)
	}
	before := map[string]string{}
	for _, c := range cases {
		anchors := c.anchors
		if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_receipts(receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,
 stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,request_digest,run_id,owner_id,project_id,
 workflow_name,workflow_version,workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,proposal_ref,proposal_digest,
 proposal_media_type,proposal_size_bytes,evidence,report_projection,anchor_projection)
 VALUES($1,$1,$1,'runtime','instance','stage','worker',$1,$1,$1,'sha256:'||repeat('a',64),'run','owner','correlation-upgrade',
 'workflow','1','contractor/v1alpha1','{}','sha256:'||repeat('b',64),'{}','sha256:'||repeat('c',64),'application/json',2,'[]','{}',$2)`, c.name, string(anchors)); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Exec(ctx, `INSERT INTO audit_findings(finding_id,audit_id,first_receipt_id,first_proposal_ref) VALUES($1,'upgrade-audit',$1,'{}')`, c.name); err != nil {
			t.Fatal(err)
		}
		var raw string
		if err := pool.QueryRow(ctx, "SELECT to_jsonb(r)::text FROM finding_proposal_receipts r WHERE receipt_id=$1", c.name).Scan(&raw); err != nil {
			t.Fatal(err)
		}
		before[c.name] = raw
	}
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		var project, algorithm, value, group, reason, receipt string
		if err := pool.QueryRow(ctx, `SELECT correlation_project_id,correlation_algorithm,correlation_value,correlation_group_key,correlation_reason,
 (SELECT (to_jsonb(r)-'comparison_projection')::text FROM finding_proposal_receipts r WHERE receipt_id=f.first_receipt_id) FROM audit_findings f WHERE finding_id=$1`, c.name).Scan(&project, &algorithm, &value, &group, &reason, &receipt); err != nil {
			t.Fatal(err)
		}
		expected := ""
		if c.algorithm != "" {
			data, err := json.Marshal([]string{"contractor/finding-group/v1", "correlation-upgrade", c.algorithm, c.value})
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(data)
			expected = hex.EncodeToString(sum[:])
		}

		if project != "correlation-upgrade" || algorithm != c.algorithm || value != c.value || reason != c.reason || group != expected || receipt != before[c.name] {
			t.Fatalf("backfill %s: %s %s %s %s %s receiptChanged=%v", c.name, project, algorithm, value, group, reason, receipt != before[c.name])
		}
	}
	if _, err := pool.Exec(ctx, "UPDATE projects SET finding_correlation_revision=0 WHERE project_id='correlation-upgrade'"); Classify(err) != ClassRejected {
		t.Fatalf("revision regressed: %v", err)
	}
	if _, err := pool.Exec(ctx, "UPDATE audit_findings SET correlation_project_id='foreign' WHERE finding_id='old'"); Classify(err) != ClassRejected {
		t.Fatalf("scope changed: %v", err)
	}
}
