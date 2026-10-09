package auditservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config"
)

// The catalog fixture is a frozen copy of the profiles, Workflows and WSTG
// standard these tests exercise. Expectations come from it, never from the
// operator-editable configs/ tree.
const (
	auditServiceCatalogFixture = "testdata/catalog"
	openAPIScanInputFixture    = "testdata/audit-openapi-scan"
)

func loadAuditServiceCatalog(t *testing.T) *config.Snapshot {
	t.Helper()
	snapshot, err := config.Load(auditServiceCatalogFixture, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func TestPreparationPreviewValidatesOriginalInputsBeforeGeneratedInventory(t *testing.T) {
	snapshot := loadAuditServiceCatalog(t)
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
	if !compatibility.ServerCompatible || len(compatibility.Reasons) != 0 {
		t.Fatalf("preparation is not available: %+v", compatibility)
	}
	if err := ValidateInputPreview(profile, Scope{}, nil, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("preview ignored required original inputs: %v", err)
	}
	inputs := map[string]artifacts.ReadResult{}
	for name, input := range profile.Inputs {
		if input.Required {
			inputs[name] = artifacts.ReadResult{Payload: artifacts.Payload{MediaType: input.MediaTypes[0], Data: []byte("original input")}}
		}
	}
	if err := ValidateInputPreview(profile, Scope{}, inputs, nil); err != nil {
		t.Fatalf("preview tried to parse inventory before preparation: %v", err)
	}
}

func TestPreparationCannotUseClassifiedToolsBeforeItemApproval(t *testing.T) {
	profile, err := loadAuditServiceCatalog(t).AuditProfile("openapi-sqlmap-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	prepare := profile.Workflows["scan"]
	prepare.Kind, prepare.MaxRunAttempts = config.AuditWorkflowPrepare, 1
	profile.Workflows["prepare-scan"] = prepare
	compatibility := ProfileCompatibility(profile)
	if compatibility.ServerCompatible || len(compatibility.Reasons) != 1 || compatibility.Reasons[0] != ReasonAutomaticActiveChecksUnsupported {
		t.Fatalf("item approval authorized preparation tools: %+v", compatibility)
	}
}
