package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestWorkflowBindingIndexMatchesCatalogFingerprintAndDetachesPages(t *testing.T) {
	snapshot := mustLoad(t, "testdata/valid", MVPDescriptors())
	index, err := snapshot.AgentTemplateWorkflowBindings("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.Configuration(ConfigurationAgentTemplates, "artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(struct {
		Template  ConfigurationRef
		Workflows []ResolvedWorkflow
	}{template.Ref, snapshot.Workflows()})
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(raw)
	if index.SourceFingerprint != hex.EncodeToString(digest[:]) {
		t.Fatal("cursor fingerprint changed")
	}
	pages := index.Page(nil, 100)
	if len(pages) == 0 {
		t.Fatal("missing template uses")
	}
	first := pages[0]
	pages[0].Stage = "modified"
	if index.Page(nil, 1)[0] != first {
		t.Fatal("page mutated immutable index")
	}
	if got := index.Page(&first, 100); !reflect.DeepEqual(got, index.Page(nil, 100)[1:]) {
		t.Fatal("cursor page skipped or repeated a use")
	}
	// A rebuilt snapshot with the same selector and a new exact revision must
	// exclude uses pinned to the previous digest, without changing old pages.
	changed := cloneAgentTemplate(snapshot.templates["artifact_builder@1"])
	changed.Ref.Digest = "sha256:changed"
	templates := make(map[string]contracts.ResolvedAgentTemplate, len(snapshot.templates))
	for key, template := range snapshot.templates {
		templates[key] = template
	}
	templates["artifact_builder@1"] = changed
	next := newSnapshot(snapshot.workflows, templates, snapshot.policies, snapshot.gateways, snapshot.executionConfigs, snapshot.auditProfiles, snapshot.instructions, snapshot.sources)
	updated, err := next.AgentTemplateWorkflowBindings("artifact_builder@1")
	if err != nil {
		t.Fatal(err)
	}
	if len(updated.Page(nil, 100)) != 0 || updated.SourceFingerprint == index.SourceFingerprint {
		t.Fatal("index mixed exact template revisions")
	}
	if index.Page(nil, 1)[0] != first {
		t.Fatal("new snapshot mutated old index")
	}
}

func TestWorkflowBindingPageAllocationsAreBounded(t *testing.T) {
	index := AgentTemplateWorkflowBindings{items: make([]AgentTemplateWorkflowBinding, 10000)}
	for i := range index.items {
		index.items[i].Workflow.Name = "workflow"
	}
	allocations := testing.AllocsPerRun(10, func() {
		if len(index.Page(nil, 2)) != 2 {
			t.Fatal("unbounded page")
		}
	})
	if allocations > 1 {
		t.Fatalf("page allocations = %f, want one bounded copy", allocations)
	}
}
