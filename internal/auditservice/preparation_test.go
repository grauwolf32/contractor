package auditservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
)

func TestPreparationIsCatalogValidButCannotStartOrPreview(t *testing.T) {
	snapshot, err := config.Load("../../configs", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("openapi-from-workspace@7")
	if err != nil {
		t.Fatal(err)
	}
	profile.Inputs["source"] = config.AuditProfileInput{Required: true, MediaTypes: []string{"application/zip"}}
	profile.Workflows["arbitrary-role"] = config.ResolvedAuditWorkflowBinding{
		Kind: config.AuditWorkflowPrepare, MaxRunAttempts: 2, Workflow: workflow,
		Inputs:     map[string]config.AuditWorkflowInputMapping{"source": {Source: config.AuditInputFromAudit, Name: "source"}},
		Parameters: map[string]config.AuditWorkflowParameterMapping{}, Outputs: map[string]string{"api": "openapi"},
	}
	if err := config.ValidateAuditPreparationProfile(profile); err != nil {
		t.Fatal(err)
	}
	compatibility := ProfileCompatibility(profile)
	if compatibility.ServerCompatible || len(compatibility.Reasons) != 1 || compatibility.Reasons[0] != ReasonPreparationUnsupported {
		t.Fatalf("prepare was advertised as runnable: %+v", compatibility)
	}
	if err := ValidateInputPreview(profile, Scope{}, nil, nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("preview attempted inventory before capability gating: %v", err)
	}
}
