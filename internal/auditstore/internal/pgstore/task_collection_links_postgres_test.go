package pgstore

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Separate Tasks may cite the same exact retained material. Link replay must
// neither collide nor charge it again; drift still rejects the whole mutation.
func TestSharedPostgresCommonTaskCollectionReusesExactEvidence(t *testing.T) {
	f := newCommonTaskFixture(t, 1, func(l *auditstore.Limits) { l.MaxEvidenceBytes = int64(len("selected source")) })
	ctx := t.Context()
	accepted, _, err := f.owners.AcceptTasks(ctx, f.acceptance(2, "shared-evidence"))
	if err != nil {
		t.Fatal(err)
	}
	claims := NewClaimStore(f.pool, artifactpgtest.NewStorage())
	claimed, err := claims.Claim(ctx, auditstore.ClaimParams{HolderID: "shared-evidence", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %+v %v", claimed, err)
	}
	claim := claimed[0]
	link := auditstore.ArtifactLink{Scope: "audit", LogicalKey: "evidence/shared", Artifact: auditstore.ExactArtifact{Ref: f.material.Ref, Digest: f.material.Digest, MediaType: f.material.MediaType, SizeBytes: f.material.SizeBytes}, SourceProvenance: []byte(`{}`)}
	for ordinal, id := range accepted.TaskIDs {
		execution, _, err := claims.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: []string{id}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := claims.ObserveSubmissionFailure(ctx, auditstore.ObserveSubmissionFailureParams{Claim: claim, ExecutionID: execution.ExecutionID}); err != nil {
			t.Fatal(err)
		}
		gap := "submission-failed"
		params := auditstore.CollectParams{Claim: claim, ExecutionID: execution.ExecutionID, ReceiptID: "receipt-" + id, RequestDigest: testDigest("7"), Disposition: auditstore.CollectionExecutionFailed, Retained: []auditstore.ArtifactLink{link}, Tasks: []auditstore.TaskCollection{{ExecutionItemID: auditdomain.DeterministicID("attempt", execution.ExecutionID, id), TaskID: id, GapCode: &gap}}}
		if ordinal == 0 {
			params.Retained[0].Scope = "project"
			if _, changed, err := claims.Collect(ctx, params); changed || !errors.Is(err, auditstore.ErrInvalid) {
				t.Fatalf("foreign material scope: changed=%v %v", changed, err)
			}
			params.Retained[0] = link
		}
		if ordinal == 1 {
			params.Retained[0].Artifact.Digest = testDigest("9")
			if _, changed, err := claims.Collect(ctx, params); changed || !errors.Is(err, auditstore.ErrInvalid) {
				t.Fatalf("drift: changed=%v err=%v", changed, err)
			}
			if _, err := f.reads.CollectionReceipt(ctx, f.audit.AuditID, execution.ExecutionID); !errors.Is(err, auditstore.ErrNotFound) {
				t.Fatalf("drift wrote a receipt: %v", err)
			}
			params.Retained[0] = link
			alias := link
			alias.LogicalKey = "evidence/second-reference"
			params.Retained = append(params.Retained, alias)
		}
		if _, changed, err := claims.Collect(ctx, params); err != nil || !changed {
			t.Fatalf("Task %d collection: %v %v", ordinal, changed, err)
		}
		current, err := f.reads.Get(ctx, f.audit.OwnerID, f.audit.AuditID)
		if err != nil || current.RetainedEvidenceBytes != f.material.SizeBytes {
			t.Fatalf("Task %d evidence bytes=%d want %d: %v", ordinal, current.RetainedEvidenceBytes, f.material.SizeBytes, err)
		}
	}
}

func TestSharedPostgresCommonTaskCollectionEvidenceLookupStaysBounded(t *testing.T) {
	ctx := t.Context()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	pool := budgetAuditPool(t, ctx, url)
	f := commonTaskFixtureOnPool(t, pool, "evidence-scale")
	ref, _ := json.Marshal(f.material.Ref)
	// Fixed read-model aliases stress the containing collection independently
	// of the bounded mutation. They all point to an actual immutable revision.
	if _, err := pool.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,artifact_scope,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance,display_ref)
 SELECT $1,'audit','evidence-'||n,$2,$3,$4,$5,'{}','' FROM generate_series(0,4999) AS n`, f.audit.AuditID, ref, f.material.Digest, f.material.MediaType, f.material.SizeBytes); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `VACUUM (INDEX_CLEANUP ON,ANALYZE) audit_artifact_links`); err != nil {
		t.Fatal(err)
	}
	params := auditstore.CollectParams{Retained: []auditstore.ArtifactLink{{Scope: "audit", LogicalKey: "evidence-0", Artifact: auditstore.ExactArtifact{Ref: f.material.Ref, Digest: f.material.Digest, MediaType: f.material.MediaType, SizeBytes: f.material.SizeBytes}, SourceProvenance: []byte(`{}`)}}}
	scope, _ := artifacts.AuditScope(f.audit.AuditID)
	fresh, err := f.service.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: "inputs", Name: "fresh-material"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected source")})
	if err != nil {
		t.Fatal(err)
	}
	newLink := params.Retained[0]
	newLink.LogicalKey = "fresh-evidence"
	newLink.Artifact.Ref = fresh.Ref
	for _, selection := range []struct {
		name  string
		links []auditstore.ArtifactLink
		cost  int64
	}{
		{"existing", params.Retained, 0},
		{"existing-and-new", append([]auditstore.ArtifactLink{newLink}, params.Retained...), f.material.SizeBytes},
	} {
		t.Run(selection.name, func(t *testing.T) {
			input, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: selection.links})
			for _, mode := range explaintest.Modes {
				plan := explainPrepared(t, ctx, pool, mode, readCollectionLinkStateSQL, f.audit.AuditID, input)
				if reads := plan.Reads.Of("audit_artifact_links"); reads > 16 {
					t.Fatalf("%s read %d link rows for bounded selected materials: %s", mode, reads, plan.Plan.String())
				}
			}
			var valid bool
			var cost int64
			if err := pool.QueryRow(ctx, readCollectionLinkStateSQL, f.audit.AuditID, input).Scan(&valid, &cost); err != nil || !valid || cost != selection.cost {
				t.Fatalf("valid=%v cost=%d want %d: %v", valid, cost, selection.cost, err)
			}
		})
	}

}

func TestCollectionCountsScopedRevisionsSeparately(t *testing.T) {
	ref := testExact("inputs", "same-path", "r1")
	ref.MediaType = "text/plain"
	ref.SizeBytes = 7
	links := []auditstore.ArtifactLink{{Scope: "audit", LogicalKey: "audit-source", Artifact: ref, SourceProvenance: []byte(`{}`)}, {Scope: "project", LogicalKey: "project-source", Artifact: ref, SourceProvenance: []byte(`{}`)}}
	validated, err := validateArtifactLinks(links)
	if err != nil || validated != 14 {
		t.Fatalf("validated scoped bytes=%d %v", validated, err)
	}
	_, prepared := prepareRetainedLinks(auditstore.CollectParams{Retained: links})
	if prepared != validated {
		t.Fatalf("prepared=%d validated=%d", prepared, validated)
	}
}
