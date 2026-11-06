package evalservice

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestSharedEvalCommonResultsStructuralChecks(t *testing.T) {
	pool := serviceTestPool(t)
	storage := artifactpgtest.NewStorage()
	scope, err := artifacts.NewService(storage.Repository(pool)).User("owner")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name     string
			Document string
			Valid    bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-common", "results.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, c := range fixture.Cases {
		if !c.Valid {
			continue
		}
		t.Run(c.Name, func(t *testing.T) {
			var doc struct{ Schema string }
			if err := json.Unmarshal([]byte(c.Document), &doc); err != nil {
				t.Fatal(err)
			}
			written, err := scope.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: c.Name}, artifacts.Payload{MediaType: "application/json", Data: []byte(c.Document)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			artifact := evaldomain.Artifact{Scope: "user", ScopeID: "owner", Namespace: written.Ref.Namespace, Name: written.Ref.Name, Revision: *written.Ref.Revision, SHA256: evaldomain.Digest([]byte(c.Document)), MediaType: "application/json", SizeBytes: int64(len(c.Document))}
			input := evaldomain.ResultInput{Outputs: map[string]evaldomain.Artifact{"result": artifact}, Evidence: []evaldomain.Evidence{{ID: "original", Artifact: artifact}}}
			checks := []evaldomain.Check{{ID: "shape", Evaluator: "json-schema@1", Required: true, ImplementationSHA256: NativePolicySHA256(), Parameters: map[string]string{"output": "result", "schema": doc.Schema}}}
			got, err := runNativeChecks(t.Context(), pool, checks, nil, input, storage)
			if err != nil || len(got) != 1 || got[0].Status != "pass" || len(got[0].EvidenceRefs) != 1 {
				t.Fatalf("structural check: %+v %v", got, err)
			}
			// A schema pass is independent of supported/refuted/inconclusive
			// and exploratory results. It grants no Finding or intake authority.
			checks[0].Parameters["schema"] = "contractor.audit.coverage.v1"
			got, err = runNativeChecks(t.Context(), pool, checks, nil, input, storage)
			if err != nil || got[0].Status != "error" {
				t.Fatalf("retired schema: %+v %v", got, err)
			}
		})
		checked++
	}
	if checked == 0 {
		t.Fatal("no common results checked")
	}
	var findings int
	if err := pool.QueryRow(t.Context(), `SELECT count(*) FROM project_findings`).Scan(&findings); err != nil || findings != 0 {
		t.Fatalf("shape checking created findings: %d %v", findings, err)
	}
}
