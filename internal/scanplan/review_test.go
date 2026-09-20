package scanplan_test

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/scanplan"
)

func TestPrepareMisspelledPathMethodReportsGap(t *testing.T) {
	doc := document(map[string]any{"/x": map[string]any{"GET": map[string]any{}}})
	view := mustDocument(t, doc, scanplan.Options{})
	if view.Coverage.Complete || len(view.Gaps) == 0 || len(view.Requests) != 0 {
		t.Fatalf("misspelled method became complete empty coverage: %+v", view)
	}
}

func TestPrepareUnknownOperationFieldWarnsAndRetainsRequest(t *testing.T) {
	doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{"paramaters": []any{map[string]any{"name": "required-value", "in": "query", "required": true}}}}})
	view := mustDocument(t, doc, scanplan.Options{})
	if len(view.Requests) != 1 || view.Coverage.Complete || len(view.Gaps) != 1 || view.Gaps[0].Code != "unsupported_operation_field" {
		t.Fatalf("unknown metadata should warn without dropping known request data: %+v", view.Coverage)
	}
}

func TestPrepareServerVariablesDoNotRecursivelyExpand(t *testing.T) {
	doc := document(map[string]any{"/x": map[string]any{"get": map[string]any{}}})
	doc["servers"] = []any{map[string]any{
		"url": "https://{host}/{prefix}",
		"variables": map[string]any{
			"host":   map[string]any{"default": "{prefix}.example.test"},
			"prefix": map[string]any{"default": "api"},
		},
	}}
	view := mustDocument(t, doc, scanplan.Options{})
	requireSkipped(t, view)
	view = mustDocument(t, doc, scanplan.Options{ServerVariables: map[string]string{"host": "{prefix}.example.test"}})
	requireSkipped(t, view)
}
