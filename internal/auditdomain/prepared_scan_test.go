package auditdomain

import (
	"bytes"
	"encoding/json"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

func scanWorkTestInput(source, settings []byte) PreparedOpenAPIInput {
	var fixture struct {
		Scanner        string
		Server         string
		CredentialID   string `json:"credential_id"`
		Operations     map[string]json.RawMessage
		TestParameters []string
	}
	if err := json.Unmarshal(settings, &fixture); err != nil {
		panic(err)
	}
	pointers, _ := json.Marshal(slices.Sorted(maps.Keys(fixture.Operations)))
	params := map[string]string{"server": fixture.Server, "operations": string(pointers), "credential_id": fixture.CredentialID}
	if fixture.TestParameters != nil {
		raw, _ := json.Marshal(fixture.TestParameters)
		params["test_parameters"] = string(raw)
	}
	values, _ := json.Marshal(fixture.Operations)
	m := preparedSource("request_values", values)
	return PreparedOpenAPIInput{Source: preparedSource("openapi", source), Data: source, Values: &m, ValuesData: values, Options: auditcontract.OpenAPIWorkArguments{Workflow: "scan", Selection: "scan-requests", MaxItems: 256, Scanner: fixture.Scanner, Parameters: params}}
}
func TestPreparedOpenAPIScanFixtureRequests(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			root := "../../testdata/configs-scan/audit-openapi-scan/"
			source, err := os.ReadFile(root + "openapi.json")
			if err != nil {
				t.Fatal(err)
			}
			parameters, err := os.ReadFile(root + scanner + "-parameters.json")
			if err != nil {
				t.Fatal(err)
			}
			values, err := os.ReadFile(root + scanner + "-request-values.json")
			if err != nil {
				t.Fatal(err)
			}
			params := map[string]string{}
			if err := json.Unmarshal(parameters, &params); err != nil {
				t.Fatal(err)
			}
			material := preparedSource("request_values", values)
			input := PreparedOpenAPIInput{Source: preparedSource("openapi", source), Data: source, Values: &material, ValuesData: values, Options: auditcontract.OpenAPIWorkArguments{Workflow: "scan", Selection: "scan-requests", MaxItems: 16, Scanner: scanner, Parameters: params}}
			work, err := BuildPreparedOpenAPIWork(t.Context(), input)
			if err != nil || len(work.Items) != 1 {
				t.Fatalf("work=%+v err=%v", work, err)
			}
			scan, prepared, err := PrepareOpenAPIScanOperation(work.Items[0].Data, source, input.ValuesData, "application/json", input.Options.Scanner, input.Options.Parameters)
			if err != nil || !prepared.Runnable || scan.Scanner != scanner {
				t.Fatalf("ordinary operation=%+v %v", prepared, err)
			}
			want := "http://127.0.0.1:8080/api/pets/7?search=Milo"
			if scanner == "nuclei" {
				if prepared.Target == nil || prepared.Target.URL != want || prepared.RequestSet != nil {
					t.Fatal("selected URL changed")
				}
			} else {
				request := prepared.RequestSet.Requests[0].Request
				if request.URL != want || request.Method != "POST" || request.Body != `{"name":"Milo"}` {
					t.Fatal("selected request changed")
				}
			}
		})
	}
}
func TestPreparedOpenAPIScanSettingsPinsGapsAndNoCredentialsInMaterial(t *testing.T) {
	input := scanWorkTestInput([]byte(scanSource), []byte(sqlmapSettings))
	work, err := BuildPreparedOpenAPIWork(t.Context(), input)
	if err != nil || len(work.Items) != 1 {
		t.Fatal(err)
	}
	if bytes.Contains(work.Items[0].Data, []byte("never-copy-credential")) {
		t.Fatal("credentials entered prepared material")
	}
	first := decodePreparedTestItem(t, work.Items[0])
	revision := "different-exact-settings"
	input.Values.Ref.Revision = &revision
	second, err := BuildPreparedOpenAPIWork(t.Context(), input)
	if err != nil || work.Items[0].Key == second.Items[0].Key {
		t.Fatal("exact settings identity was lost")
	}
	if first.Scan == nil || first.Values == nil || first.Values.Digest != input.Values.Digest {
		t.Fatal("settings digest missing")
	}
	input = scanWorkTestInput([]byte(scanSource), []byte(nucleiSettings))
	work, err = BuildPreparedOpenAPIWork(t.Context(), input)
	if err != nil || len(work.Items) != 2 {
		t.Fatal(err)
	}
	missing := 0
	for _, item := range work.Items {
		d := decodePreparedTestItem(t, item)
		if d.Scan == nil {
			t.Fatal("selected scan missing")
		}
		if !d.Scan.Runnable {
			missing++
			if len(d.Scan.Gaps) == 0 {
				t.Fatal("incomplete request lost gaps")
			}
		}
	}
	if missing != 1 {
		t.Fatal("missing concrete parameter became runnable")
	}
}
func TestPreparedOpenAPIScanRepreparationRejectsChangedBytes(t *testing.T) {
	input := scanWorkTestInput([]byte(scanSource), []byte(sqlmapSettings))
	work, err := BuildPreparedOpenAPIWork(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	for _, alter := range []string{"source", "settings", "material"} {
		source, settings, item := input.Data, input.ValuesData, work.Items[0].Data
		switch alter {
		case "source":
			source = append(append([]byte{}, source...), ' ')
		case "settings":
			settings = append(append([]byte{}, settings...), ' ')
		case "material":
			var d PreparedOpenAPIOperation
			if err := json.Unmarshal(item, &d); err != nil {
				t.Fatal(err)
			}
			d.Scan.PreparationDigest = strings.Repeat("x", 71)
			item, _ = json.Marshal(d)
		}
		if _, _, err := PrepareOpenAPIScanOperation(item, source, settings, "application/json", input.Options.Scanner, input.Options.Parameters); err == nil {
			t.Fatalf("%s changed trusted exact selection", alter)
		}
	}
	invalid := scanWorkTestInput([]byte(scanSource), []byte(strings.Replace(sqlmapSettings, "#/paths/~1pets~1{id}/post", "#/paths/~1foreign/get", 1)))
	if work, err := BuildPreparedOpenAPIWork(t.Context(), invalid); err == nil || len(work.Items) != 0 {
		t.Fatal("malformed selection partially accepted")
	}
}
