package runservice

import (
	"testing"

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
			"sha256:c80ac1684eebe6e1cc407c2bebed94694a3fdea58f74a2aa8370253ff1dbad42"},
		{"project run", PublicCreateParams{Workflow: "scan", ProjectID: &projectID},
			"sha256:5c327e88b95acdbd378c326188e609eb934fc2157fcd25039dcfddde37fcb668"},
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
	ordered := digest(PublicCreateParams{Workflow: "empty@1", RuntimeLabels: []string{"debug", "trace"}})
	reversed := digest(PublicCreateParams{Workflow: "empty@1", RuntimeLabels: []string{"trace", "debug"}})
	if reversed != ordered || reversed == omitted {
		t.Fatalf("label digests = ordered:%q reversed:%q omitted:%q", ordered, reversed, omitted)
	}
	placement := digest(PublicCreateParams{Workflow: "empty@1", MetadataLabels: map[string]string{"cloud": "aws", "team": "appsec"}})
	placementReversed := digest(PublicCreateParams{Workflow: "empty@1", MetadataLabels: map[string]string{"team": "appsec", "cloud": "aws"}})
	otherPlacement := digest(PublicCreateParams{Workflow: "empty@1", MetadataLabels: map[string]string{"local": "lab"}})
	if placement != placementReversed || placement == omitted || placement == otherPlacement {
		t.Fatal("placement requirements are not canonically bound to the request digest")
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
