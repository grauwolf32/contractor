package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"testing"
)

func catalogSourceDigest(t *testing.T, source any) string {
	t.Helper()
	encoded, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:])
}

// TestDiscoveryFingerprintsTheCatalogOnceAndPagesDetachedCopies holds the
// discovery views to the fingerprint the cursor binding always used (CAT-R67)
// while a page copies only the Workflows it returns.
func TestDiscoveryFingerprintsTheCatalogOnceAndPagesDetachedCopies(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t, "workflow-examples"))
	workflows := snapshot.Workflows()
	if len(workflows) < 3 {
		t.Fatalf("fixture snapshot has only %d Workflows", len(workflows))
	}
	discovery, err := snapshot.WorkflowDiscovery()
	if err != nil || discovery.SourceFingerprint != catalogSourceDigest(t, workflows) {
		t.Fatalf("Workflow fingerprint = (%q, %v), want SHA-256 of the sorted Workflows", discovery.SourceFingerprint, err)
	}
	for _, kind := range configurationKinds {
		configurations, err := snapshot.ConfigurationDiscovery(kind)
		if err != nil || len(configurations.Resources) == 0 ||
			configurations.SourceFingerprint != catalogSourceDigest(t, configurations.Resources) {
			t.Fatalf("%s discovery = (%d resources, %q, %v)", kind, len(configurations.Resources),
				configurations.SourceFingerprint, err)
		}
	}
	if _, err := snapshot.ConfigurationDiscovery("workflows"); err == nil {
		t.Fatal("an unknown configuration kind was discovered")
	}

	all := func(WorkflowRef, WorkflowPresentation) bool { return true }
	first := discovery.Page("", 2, all)
	if !reflect.DeepEqual(first, workflows[:2]) {
		t.Fatalf("first page = %v, want the first two Workflows", workflowSelectors(first))
	}
	after := workflows[0].Ref.Name + "@" + workflows[0].Ref.Version
	if next := discovery.Page(after, 1, all); !reflect.DeepEqual(next, workflows[1:2]) {
		t.Fatalf("page after %s = %v", after, workflowSelectors(next))
	}
	last := workflows[len(workflows)-1]
	if rest := discovery.Page(last.Ref.Name+"@"+last.Ref.Version, 5, all); len(rest) != 0 {
		t.Fatalf("page after the last Workflow = %v", workflowSelectors(rest))
	}
	matched := discovery.Page("", len(workflows), func(ref WorkflowRef, _ WorkflowPresentation) bool {
		return ref == workflows[2].Ref
	})
	if !reflect.DeepEqual(matched, workflows[2:3]) {
		t.Fatalf("filtered page = %v, want only %s", workflowSelectors(matched), workflows[2].Ref.Name)
	}

	first[0].Parameters["tampered"] = ParameterSlot{Required: true}
	first[0].Stages[first[0].EntryStage] = ResolvedStage{Objective: "tampered"}
	again, err := snapshot.WorkflowDiscovery()
	if err != nil {
		t.Fatal(err)
	}
	if reread := again.Page("", 1, all); !reflect.DeepEqual(reread, workflows[:1]) {
		t.Fatal("a page caller's edits reached the immutable Workflow snapshot")
	}
}

func workflowSelectors(workflows []ResolvedWorkflow) []string {
	selectors := make([]string, 0, len(workflows))
	for _, workflow := range workflows {
		selectors = append(selectors, workflow.Ref.Name+"@"+workflow.Ref.Version)
	}
	return selectors
}
