package runstore

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestRunMetadataLabelsNormalizeCloneAndSort(t *testing.T) {
	t.Parallel()
	source := map[string]string{
		"purpose":     "eval",
		"eval.name":   "openapi regression",
		"eval-case_1": "fixture/path?a=1",
	}
	labels, err := NormalizeRunMetadataLabels(source)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := labels.SortedKeys(), []string{"eval-case_1", "eval.name", "purpose"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("sorted keys = %v, want %v", got, want)
	}
	delete(source, "purpose")
	if labels["purpose"] != "eval" {
		t.Fatal("normalized labels retained caller map alias")
	}
	clone := labels.Clone()
	clone["purpose"] = "changed"
	if labels["purpose"] != "eval" {
		t.Fatal("Clone retained source map alias")
	}
	empty, err := NormalizeRunMetadataLabels(nil)
	if err != nil || empty == nil || len(empty) != 0 {
		t.Fatalf("nil normalization = %#v, %v", empty, err)
	}
}

func TestRunMetadataLabelsRejectInvalidBoundsAndShapes(t *testing.T) {
	t.Parallel()
	tests := []map[string]string{
		{"Upper": "value"},
		{"two..parts": "value"},
		{"trailing.": "value"},
		{"contractor.internal": "value"},
		{"purpose": ""},
		{strings.Repeat("a", MaxRunMetadataLabelKey+1): "value"},
		{"purpose": strings.Repeat("Ж", MaxRunMetadataLabelValue/2+1)},
		{"purpose": string([]byte{0xff})},
		{"purpose": "before\x00after"},
	}
	tooMany := make(map[string]string)
	for index := 0; index <= MaxRunMetadataLabels; index++ {
		tooMany["key"+string(rune('a'+index%26))+strings.Repeat("x", index/26)] = "value"
	}
	tests = append(tests, tooMany)
	for _, labels := range tests {
		if _, err := NormalizeRunMetadataLabels(labels); !errors.Is(err, ErrInvalid) {
			t.Fatalf("NormalizeRunMetadataLabels(%q) error = %v", labels, err)
		}
	}
}

func TestRunMetadataLabelSelectorsNormalizeWithoutHidingContradictions(t *testing.T) {
	t.Parallel()
	selectors, err := NormalizeRunMetadataLabelSelectors([]RunMetadataLabelSelector{
		{Key: "purpose", Value: "eval"},
		{Key: "eval.leg", Value: "b"},
		{Key: "purpose", Value: "eval"},
		{Key: "eval.leg", Value: "a"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []RunMetadataLabelSelector{
		{Key: "eval.leg", Value: "a"},
		{Key: "eval.leg", Value: "b"},
		{Key: "purpose", Value: "eval"},
	}
	if !reflect.DeepEqual(selectors, want) {
		t.Fatalf("normalized selectors = %+v, want %+v", selectors, want)
	}
	tooMany := make([]RunMetadataLabelSelector, MaxRunMetadataLabels+1)
	for index := range tooMany {
		tooMany[index] = RunMetadataLabelSelector{Key: "purpose", Value: "eval"}
	}
	if _, err := NormalizeRunMetadataLabelSelectors(tooMany); !errors.Is(err, ErrInvalid) {
		t.Fatalf("over-limit duplicate selectors error = %v", err)
	}
}
