package auditdomain

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

func scanWorkTestInput(source, settings []byte) PreparedOpenAPIInput {
	m := preparedSource("settings", settings)
	return PreparedOpenAPIInput{Source: preparedSource("openapi", source), Data: source, Settings: &m, SettingsData: settings, Options: auditcontract.OpenAPIWorkArguments{Workflow: "scan", Selection: "scan-requests", MaxItems: 256}}
}
func TestPreparedOpenAPIScanFixtureRequests(t *testing.T) {
	for _, scanner := range []string{"sqlmap", "nuclei"} {
		t.Run(scanner, func(t *testing.T) {
			root := "../../testdata/configs-scan/audit-openapi-scan/"
			source, err := os.ReadFile(root + "openapi.json")
			if err != nil {
				t.Fatal(err)
			}
			settings, err := os.ReadFile(root + scanner + "-settings.json")
			if err != nil {
				t.Fatal(err)
			}
			input := scanWorkTestInput(source, settings)
			work, err := BuildPreparedOpenAPIWork(t.Context(), input)
			if err != nil || len(work.Items) != 1 {
				t.Fatalf("work=%+v err=%v", work, err)
			}
			scan, prepared, err := PrepareOpenAPIScanOperation(work.Items[0].Data, source, settings, "application/json")
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
	input.Settings.Ref.Revision = &revision
	second, err := BuildPreparedOpenAPIWork(t.Context(), input)
	if err != nil || work.Items[0].Key == second.Items[0].Key {
		t.Fatal("exact settings identity was lost")
	}
	if first.Scan == nil || first.Scan.Settings.Digest != input.Settings.Digest {
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
		source, settings, item := input.Data, input.SettingsData, work.Items[0].Data
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
		if _, _, err := PrepareOpenAPIScanOperation(item, source, settings, "application/json"); err == nil {
			t.Fatalf("%s changed trusted exact selection", alter)
		}
	}
	invalid := scanWorkTestInput([]byte(scanSource), []byte(strings.Replace(sqlmapSettings, "#/paths/~1pets~1{id}/post", "#/paths/~1foreign/get", 1)))
	if work, err := BuildPreparedOpenAPIWork(t.Context(), invalid); err == nil || len(work.Items) != 0 {
		t.Fatal("malformed selection partially accepted")
	}
}
