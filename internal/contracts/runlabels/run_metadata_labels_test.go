package runlabels

import "testing"

func TestNormalizeRunMetadataLabelsDetachesSource(t *testing.T) {
	t.Parallel()

	source := map[string]string{"eval.id": "eval_01"}
	normalized, err := NormalizeRunMetadataLabels(source)
	if err != nil {
		t.Fatal(err)
	}
	source["eval.id"] = "changed"
	if normalized["eval.id"] != "eval_01" || normalized.Clone() == nil {
		t.Fatalf("normalized labels alias source or lost explicit empty semantics: %v", normalized)
	}
}
