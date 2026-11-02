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
			options := testInventoryOptions("check")
			options.SourceInputName = "openapi"
			options.SourceRef = input.Source.Ref
			options.ApprovalRequirement = ApprovalNone
			var original Inventory
			var err error
			if selection == "scan-requests" {
				settings := []byte(nucleiSettings)
				material := preparedSource("settings", settings)
				input.Settings = &material
				input.SettingsData = settings
				original, err = BuildOpenAPIScanInventory(data, "application/json", settings, ExactInput{Name: "settings", Ref: material.Ref, Digest: material.Digest}, options)
			} else {
				original, err = BuildOpenAPIInventory(data, "application/json", options)
			}
			if err != nil {
				t.Fatal(err)
			}
			work, err := BuildPreparedOpenAPIWork(t.Context(), input)
			if err != nil {
				t.Fatal(err)
			}
			if len(work.Items) != len(original.Tasks) || !reflect.DeepEqual(work.Gaps, original.Gaps) {
				t.Fatal("domain selection/gaps changed")
			}
			for i, item := range work.Items {
				if item.Key != original.Tasks[i].Item.ItemKey || json.Valid(item.Data) == false {
					t.Fatal("operation identity/order changed")
				}
				var material struct {
					Schema    string
					Source    auditcontract.Material
					Settings  *auditcontract.Material
					Operation *OperationTask
					Scan      *ScanTask
				}
				if err := json.Unmarshal(item.Data, &material); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(material.Operation, original.Tasks[i].Document.Operation) || !reflect.DeepEqual(material.Scan, original.Tasks[i].Document.Scan) {
					t.Fatal("operation/request conditions changed")
				}
			}
		})
	}
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
