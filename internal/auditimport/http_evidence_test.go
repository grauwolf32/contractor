package auditimport

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Inject only read failures, never emulate Artifact validation or persistence.
// Successful publication/retention runs against PostgreSQL in the process gate.
type unavailableHTTPArtifacts struct {
	ArtifactAccess
	err error
}

func (a unavailableHTTPArtifacts) DescribeRunBinding(context.Context, string, contracts.ArtifactRef) (auditstore.ExactArtifact, bool, error) {
	return auditstore.ExactArtifact{}, false, a.err
}

func TestSelectedHTTPReferenceRejectionIsPermanentButOutageIsNot(t *testing.T) {
	run, revision := "run-1", "exact"
	material := contracts.MaterialDescriptor{ScopeKind: "run", ScopeID: run,
		Ref:    contracts.ArtifactRef{Namespace: "http-exchanges", Name: "selected", Revision: &revision},
		Digest: "sha256:" + strings.Repeat("a", 64), MediaType: auditdomain.HTTPExchangeMediaType, SizeBytes: 3}
	raw, err := strictjson.Canonical(map[string]any{"schema": "contractor.audit.conclusions.v1", "items": []any{map[string]any{
		"item_key": "a", "rationale": "Observed outcome", "tested_conditions": []string{}, "limitations": []string{}, "evidence": []any{material},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	trusted := []byte(`{"items":[{"item_key":"a","evidence":[]}]}`)
	for _, readErr := range []error{artifacts.ErrArtifactNotFound, artifacts.ErrExactRevisionRequired, blob.ErrIntegrity, errors.New("store unavailable"), context.Canceled} {
		t.Run(readErr.Error(), func(t *testing.T) {
			i := &Importer{artifacts: unavailableHTTPArtifacts{err: readErr}}
			_, _, err := i.taskHTTPContext(t.Context(), auditstore.Audit{}, auditstore.Execution{RunID: &run}, trusted, raw)
			if artifacts.IsExactReferenceRejection(readErr) {
				if !errors.Is(err, ErrPermanent) {
					t.Fatalf("invalid reference must settle permanently: %v", err)
				}
			} else if !errors.Is(err, readErr) {
				t.Fatalf("infrastructure/cancellation must remain unverified: %v", err)
			}
		})
	}
}
