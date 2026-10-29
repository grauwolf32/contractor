package controlplane

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
)

func TestSitemapGrantPinsSelectedOperationsAndNeverAliasesPolicy(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("agent-sitemap")
	registration.SupportedToolsets = append(registration.SupportedToolsets, control.ToolsetCapability{Ref: "sitemap@1", Tools: contracts.SitemapToolNames()})
	registerReadyWith(t, registry, registration)
	template := testTemplate(t)
	template.Toolsets = append(template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "sitemap", Version: "1"}, Tools: []string{contracts.SitemapListTool}})
	deployment := "deployment"
	template.Sitemap = &contracts.ProjectSitemapPolicy{Version: contracts.SitemapAccessVersion, Deployment: &deployment, SnapshotInputs: []string{"api"}}
	reservations, err := registry.ReserveAll(ReservationRequest{RunID: "run-sitemap", StageExecutionID: "stage-sitemap", Bindings: []BindingRequirement{testBinding(t, "reader", "research", template)}})
	if err != nil || len(reservations) != 1 {
		t.Fatalf("sitemap reservation: %+v %v", reservations, err)
	}
	id := reservations[0].Grant.AllocationID
	if targets := reservations[0].Grant.ProjectArtifactReads; len(targets) != 1 || targets[0].Namespace != "sitemap" || targets[0].Name != "map" || len(targets[0].Operations) != 1 || targets[0].Operations[0] != contracts.SitemapListTool {
		t.Fatalf("pinned Project reads: %+v", targets)
	}
	// Both exported reservation and later grant copies are untrusted aliases.
	reservations[0].Grant.SitemapTools[0] = contracts.SitemapRemoveTool
	reservations[0].Grant.SitemapPolicy.CollectObservations = true
	*reservations[0].Grant.SitemapPolicy.Deployment = "other"
	reservations[0].Grant.SitemapPolicy.SnapshotInputs[0] = "other"
	reservations[0].Grant.ProjectArtifactReads[0].Name = "other"
	reservations[0].Grant.ProjectArtifactReads[0].Operations[0] = "other"
	grant, err := registry.GetGrant(id)
	if err != nil || len(grant.SitemapTools) != 1 || grant.SitemapTools[0] != contracts.SitemapListTool || grant.SitemapPolicy.CollectObservations || *grant.SitemapPolicy.Deployment != "deployment" || grant.SitemapPolicy.SnapshotInputs[0] != "api" {
		t.Fatalf("grant authority aliased reservation: %+v %v", grant, err)
	}
	grant.SitemapTools[0] = contracts.SitemapUpsertTool
	if grant.ProjectArtifactReads[0].Name != "map" || grant.ProjectArtifactReads[0].Operations[0] != contracts.SitemapListTool {
		t.Fatal("Project read target aliased reservation")
	}
	grant.ProjectArtifactReads[0].Operations[0] = "other"
	*grant.SitemapPolicy.Deployment = "other"
	if err := registry.WithWriteGrant(id, func(copy AllocationGrant) error {
		if *copy.SitemapPolicy.Deployment != "deployment" || copy.SitemapTools[0] != contracts.SitemapListTool {
			t.Fatal("grant getter mutated stored policy")
		}
		*copy.SitemapPolicy.Deployment = "other"
		copy.SitemapTools[0] = contracts.SitemapRestoreTool
		if copy.ProjectArtifactReads[0].Operations[0] != contracts.SitemapListTool {
			t.Fatal("Project read target aliased getter")
		}
		copy.ProjectArtifactReads[0].Operations[0] = "other"
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if grant, err := registry.GetGrant(id); err != nil || *grant.SitemapPolicy.Deployment != "deployment" || grant.SitemapTools[0] != contracts.SitemapListTool || grant.ProjectArtifactReads[0].Operations[0] != contracts.SitemapListTool {
		t.Fatalf("callback mutated stored grant: %+v %v", grant, err)
	}
	if err := registry.SetWriteFence(id); err != nil {
		t.Fatal(err)
	}
	if grant, err := registry.GetGrant(id); err != nil || !grant.WriteFenced {
		t.Fatal("sitemap grant escaped common fence")
	}
	if err := registry.Release(id); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.GetGrant(id); !errors.Is(err, ErrAllocationNotFound) {
		t.Fatal("released sitemap grant retained authority")
	}
}
