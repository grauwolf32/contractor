package auditdomain

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedOpenAPIYAMLScalarCasesMatchRuntimeAuthoring holds the
// openapi-operations@1 inventory reader to the YAML scalar and key rule
// (SCN-R124, AUD-R143) that scan preparation and the Runtime OpenAPI
// authoring Toolset apply to the same documents, including the scalars
// openapi@1 writes.
func TestSharedOpenAPIYAMLScalarCasesMatchRuntimeAuthoring(t *testing.T) {
	cases := contractstest.ReadOpenAPIYAMLCases(t)
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			document, err := parseJSONOrYAML(tc.Document, "application/yaml")
			if !tc.Accepted {
				var failure *ValidationError
				if !errors.As(err, &failure) || failure.Code != CodeInvalid {
					t.Fatalf("parseJSONOrYAML(%q) = %v, %v; want %s", tc.Document, document, err, CodeInvalid)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseJSONOrYAML(%q) rejected an accepted case: %v", tc.Document, err)
			}
			if !contractstest.SameJSONValue(map[string]any(document), tc.Value) {
				t.Fatalf("parseJSONOrYAML(%q) = %#v, want %#v", tc.Document, document, tc.Value)
			}
		})
	}
}

func TestSharedOpenAPIJSONEscapeCasesMatchRuntimeAuthoring(t *testing.T) {
	for _, tc := range contractstest.ReadOpenAPIJSONCases(t) {
		t.Run(tc.Name, func(t *testing.T) {
			document, err := parseJSONOrYAML([]byte(tc.Document), tc.MediaType)
			if !tc.Accepted {
				var failure *ValidationError
				if !errors.As(err, &failure) || failure.Code != CodeInvalid {
					t.Fatalf("rejected JSON case = %v, %v; want %s", document, err, CodeInvalid)
				}
				return
			}
			if err != nil || !contractstest.SameJSONValue(map[string]any(document), tc.Value) {
				t.Fatalf("accepted JSON case = %#v, %v; want %#v", document, err, tc.Value)
			}
		})
	}
}
