package auditdomain

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

func preparedSource(name string, data []byte) auditcontract.Material {
	revision := "exact-" + name
	return auditcontract.Material{ScopeKind: "run", ScopeID: "producer", Ref: contracts.ArtifactRef{Namespace: "inputs", Name: name, Revision: &revision}, Digest: DigestBytes(data), MediaType: "application/json", SizeBytes: int64(len(data))}
}

func TestPreparedOpenAPIPreservesDomainSelectionWithoutPackages(t *testing.T) {
	for _, selection := range []string{"operations", "scan-requests"} {
		t.Run(selection, func(t *testing.T) {
			data := []byte(scanSource)
			input := PreparedOpenAPIInput{Source: preparedSource("openapi", data), Data: data, Options: auditcontract.OpenAPIWorkArguments{Workflow: "check", Selection: selection, MaxItems: 256}}
			expected := 3
			if selection == "scan-requests" {
				input = scanWorkTestInput(data, []byte(nucleiSettings))
				expected = 2
			}
			work, err := BuildPreparedOpenAPIWork(t.Context(), input)
			if err != nil || len(work.Items) != expected {
				t.Fatalf("selected work = %+v, %v", work, err)
			}
			seen := map[string]bool{}
			for _, item := range work.Items {
				decoded := decodePreparedTestItem(t, item)
				if item.Key == "" || seen[item.Key] || decoded.Schema != "contractor.audit.openapi-work-item.v1" || !reflect.DeepEqual(decoded.Source, input.Source) {
					t.Fatal("identity/exact provenance changed")
				}
				seen[item.Key] = true
				if selection == "operations" {
					if decoded.Operation == nil || decoded.Scan != nil {
						t.Fatal("missing operation")
					}
				} else {
					if decoded.Scan == nil || decoded.Operation != nil || !reflect.DeepEqual(decoded.Values, input.Values) {
						t.Fatal("settings selection changed")
					}
					_, prepared, err := PrepareOpenAPIScanOperation(item.Data, data, input.ValuesData, input.Source.MediaType, input.Options.Scanner, input.Options.Parameters)
					if err != nil || prepared.Operation != decoded.Scan.Operation {
						t.Fatalf("ordinary scan re-preparation = %v", err)
					}
				}
			}
		})
	}
}

func prepareTestWork(t testing.TB, data []byte, media string) (PreparedOpenAPIWork, error) {
	t.Helper()
	source := preparedSource("openapi", data)
	source.MediaType = media
	return BuildPreparedOpenAPIWork(t.Context(), PreparedOpenAPIInput{Source: source, Data: data, Options: auditcontract.OpenAPIWorkArguments{Workflow: "check", Selection: "operations", MaxItems: 256}})
}
func decodePreparedTestItem(t testing.TB, item PreparedOpenAPIItem) PreparedOpenAPIOperation {
	t.Helper()
	var result PreparedOpenAPIOperation
	if err := json.Unmarshal(item.Data, &result); err != nil {
		t.Fatal(err)
	}
	return result
}
func testDomainErrorCode(err error) string {
	var v *ValidationError
	if errors.As(err, &v) {
		return v.Code
	}
	return ""
}

func TestPreparedOpenAPIBoundsAndCancellation(t *testing.T) {
	for _, count := range []int{256, 257} {
		paths := map[string]any{}
		for i := 0; i < count; i++ {
			paths[fmt.Sprintf("/operation-%03d", i)] = map[string]any{"get": map[string]any{}}
		}
		data, _ := json.Marshal(map[string]any{"openapi": "3.1.0", "paths": paths})
		input := PreparedOpenAPIInput{Source: preparedSource("openapi", data), Data: data, Options: auditcontract.OpenAPIWorkArguments{Workflow: "check", Selection: "operations", MaxItems: 256}}
		work, err := BuildPreparedOpenAPIWork(t.Context(), input)
		if count == 256 {
			if err != nil || len(work.Items) != count {
				t.Fatalf("at ceiling: %d %v", len(work.Items), err)
			}
		} else if err == nil || len(work.Items) != 0 {
			t.Fatal("overflow returned partial work")
		}
	}
	data := []byte(`{"openapi":"3.1.0","paths":{"/large":{"get":{"description":"` + strings.Repeat("x", 70000) + `"}}}}`)
	input := PreparedOpenAPIInput{Source: preparedSource("openapi", data), Data: data, Options: auditcontract.OpenAPIWorkArguments{Workflow: "check", Selection: "operations", MaxItems: 256}}
	work, err := BuildPreparedOpenAPIWork(t.Context(), input)
	if err != nil || len(work.Items) != 1 {
		t.Fatalf("valid long source text lost: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if work, err := BuildPreparedOpenAPIWork(ctx, input); !errors.Is(err, context.Canceled) || len(work.Items) != 0 {
		t.Fatal("cancelled work escaped")
	}
	input.Source.Digest = "sha256:" + strings.Repeat("0", 64)
	if work, err := BuildPreparedOpenAPIWork(t.Context(), input); err == nil || len(work.Items) != 0 {
		t.Fatal("changed selected bytes accepted")
	}
}
