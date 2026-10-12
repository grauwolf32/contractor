package runservice

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// TestPublicRequestDigestIsPinned pins the canonical encoding a public Run
// creation key binds, so a change to it is deliberate rather than a side
// effect of a refactoring: a changed digest turns every retry of a committed
// request into an idempotency conflict.
func TestPublicRequestDigestIsPinned(t *testing.T) {
	projectID := "project-1"
	for _, test := range []struct {
		name   string
		params PublicCreateParams
		want   string
	}{
		{"run", PublicCreateParams{Workflow: "scan", RuntimeLabels: []string{"linux"}},
			"sha256:1b762ef9ccd8b775816c9e4eec1ce46c60d08115647966ac58cccc40aecb711a"},
		{"project run", PublicCreateParams{Workflow: "scan", ProjectID: &projectID},
			"sha256:558a08c1a75206f4c99e8318fcad51e411d581b4fbf87b4614806a31299d05b9"},
	} {
		if got, err := PublicRequestDigest(test.params); err != nil || got != test.want {
			t.Errorf("%s digest = %s, %v; want %s", test.name, got, err, test.want)
		}
	}
}

func TestPublicRequestDigestNormalizesEquivalentRequests(t *testing.T) {
	digest := func(params PublicCreateParams) string {
		t.Helper()
		value, err := PublicRequestDigest(params)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	omitted := digest(PublicCreateParams{Workflow: "empty@1"})
	if explicit := digest(PublicCreateParams{
		Workflow: "empty@1", Parameters: map[string]string{}, Inputs: map[string]contracts.ArtifactRef{},
	}); explicit != omitted {
		t.Fatalf("empty mappings digest = %q, omitted = %q", explicit, omitted)
	}
	if explicit := digest(PublicCreateParams{Workflow: "empty@1", MetadataLabels: runstore.RunMetadataLabels{}}); explicit != omitted {
		t.Fatalf("empty metadata label digest = %q, omitted = %q", explicit, omitted)
	}
	var emptyExecutionConfig resolved.ExecutionConfigPatch
	if err := json.Unmarshal([]byte(`{"stages":{}}`), &emptyExecutionConfig); err != nil {
		t.Fatal(err)
	}
	if explicit := digest(PublicCreateParams{Workflow: "empty@1", ExecutionConfig: emptyExecutionConfig}); explicit != omitted {
		t.Fatalf("empty executionConfig digest = %q, omitted = %q", explicit, omitted)
	}
	ordered := digest(PublicCreateParams{Workflow: "empty@1", RuntimeLabels: []string{"debug", "trace"}})
	reversed := digest(PublicCreateParams{Workflow: "empty@1", RuntimeLabels: []string{"trace", "debug"}})
	if reversed != ordered || reversed == omitted {
		t.Fatalf("label digests = ordered:%q reversed:%q omitted:%q", ordered, reversed, omitted)
	}
	metadataOrdered := digest(PublicCreateParams{
		Workflow: "empty@1", MetadataLabels: runstore.RunMetadataLabels{"purpose": "eval", "eval.leg": "a"},
	})
	metadataReversed := digest(PublicCreateParams{
		Workflow: "empty@1", MetadataLabels: runstore.RunMetadataLabels{"eval.leg": "a", "purpose": "eval"},
	})
	if metadataReversed != metadataOrdered || metadataReversed == omitted {
		t.Fatalf("metadata label digests = ordered:%q reversed:%q omitted:%q", metadataOrdered, metadataReversed, omitted)
	}
}

// A key bound to a standalone Run cannot replay into a Project, nor a key of
// one Project into another.
func TestPublicRequestDigestBindsTheSourceScope(t *testing.T) {
	first, second := "project-one", "project-two"
	seen := map[string]string{}
	for name, projectID := range map[string]*string{"standalone": nil, first: &first, second: &second} {
		value, err := PublicRequestDigest(PublicCreateParams{Workflow: "artifact-copy@1", ProjectID: projectID})
		if err != nil {
			t.Fatal(err)
		}
		if other, found := seen[value]; found {
			t.Fatalf("%s and %s share the request digest %s", name, other, value)
		}
		seen[value] = name
	}
}

func TestPublicRequestDigestRejectsInvalidLabels(t *testing.T) {
	if _, err := PublicRequestDigest(PublicCreateParams{Workflow: "scan", RuntimeLabels: []string{"Not A Label"}}); err == nil {
		t.Fatal("an invalid Runtime label was digested")
	}
	if _, err := PublicRequestDigest(PublicCreateParams{
		Workflow: "scan", MetadataLabels: runstore.RunMetadataLabels{"Upper": "value"},
	}); err == nil {
		t.Fatal("an invalid metadata label was digested")
	}
}
