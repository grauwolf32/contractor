package scanplan

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedOpenAPIYAMLScalarCasesMatchRuntimeAuthoring holds scan
// preparation to the YAML scalar and key rule (SCN-R124) that the Audit
// inventory and the Runtime OpenAPI authoring Toolset apply to the same
// documents, including the scalars openapi@1 writes.
func TestSharedOpenAPIYAMLScalarCasesMatchRuntimeAuthoring(t *testing.T) {
	cases := contractstest.ReadOpenAPIYAMLCases(t)
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			document, err := parseDocument(tc.Document, "application/yaml")
			if !tc.Accepted {
				var failure *PreparationError
				if !errors.As(err, &failure) || failure.Code != "invalid_document" {
					t.Fatalf("parseDocument(%q) = %v, %v; want invalid_document", tc.Document, document, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseDocument(%q) rejected an accepted case: %v", tc.Document, err)
			}
			if !contractstest.SameJSONValue(map[string]any(document), tc.Value) {
				t.Fatalf("parseDocument(%q) = %#v, want %#v", tc.Document, document, tc.Value)
			}
		})
	}
}

func TestSharedOpenAPIJSONEscapeCasesMatchRuntimeAuthoring(t *testing.T) {
	for _, tc := range contractstest.ReadOpenAPIJSONCases(t) {
		t.Run(tc.Name, func(t *testing.T) {
			document, err := parseDocument([]byte(tc.Document), tc.MediaType)
			if !tc.Accepted {
				var failure *PreparationError
				if !errors.As(err, &failure) || failure.Code != "invalid_document" {
					t.Fatalf("rejected JSON case = %v, %v; want invalid_document", document, err)
				}
				return
			}
			if err != nil || !contractstest.SameJSONValue(map[string]any(document), tc.Value) {
				t.Fatalf("accepted JSON case = %#v, %v; want %#v", document, err, tc.Value)
			}
		})
	}
}
