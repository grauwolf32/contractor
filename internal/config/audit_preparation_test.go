package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAuditPreparationAuthoringFixtures(t *testing.T) {
	fixture := "../../api/testdata/audit-composition"
	profileYAML := string(readFile(t, filepath.Join(fixture, "prepared-openapi-scan.yaml")))
	var cases []struct {
		Name         string      `json:"name"`
		Valid        bool        `json:"valid"`
		Replacements [][2]string `json:"replacements"`
	}
	if err := json.Unmarshal(readFile(t, filepath.Join(fixture, "profile-cases.json")), &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			root := copyConfigTree(t)
			data := profileYAML
			for _, replacement := range tc.Replacements {
				if !strings.Contains(data, replacement[0]) {
					t.Fatalf("fixture replacement does not match: %s", replacement[0])
				}
				data = strings.ReplaceAll(data, replacement[0], replacement[1])
			}
			writeAuditProfile(t, root, "prepared-openapi-scan", data)
			snapshot, err := Load(root, MVPDescriptors())
			if !tc.Valid {
				if err == nil {
					t.Fatal("invalid preparation profile accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			profile, err := snapshot.AuditProfile("prepared-openapi-scan@1")
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(profile)
			if err != nil {
				t.Fatal(err)
			}
			restored, err := DecodeResolvedAuditProfileSnapshot(encoded)
			if err != nil || !reflect.DeepEqual(profile, restored) {
				t.Fatalf("snapshot round trip: %v", err)
			}
			profile.Inventory.Source.Name = "mutated"
			again, err := snapshot.AuditProfile("prepared-openapi-scan@1")
			if err != nil || again.Inventory.Source.Name != "api" {
				t.Fatal("inventory source mutation leaked")
			}
		})
	}
}

func TestAuditPreparationDependenciesAndSnapshotValidation(t *testing.T) {
	root := copyConfigTree(t)
	data := string(readFile(t, "../../api/testdata/audit-composition/prepared-openapi-scan.yaml"))
	writeAuditProfile(t, root, "prepared-openapi-scan", data)
	snapshot := mustLoad(t, root, MVPDescriptors())
	profile, err := snapshot.AuditProfile("prepared-openapi-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	second := profile.Workflows["generate-api"]
	second.Inputs = cloneMap(second.Inputs)
	second.Inputs["existing_openapi"] = AuditWorkflowInputMapping{Source: AuditInputFromPreparation, Role: "generate-api", Name: "api"}
	profile.Workflows["refine-api"] = second
	profile.Inventory.Source.Role = "refine-api"
	profile.Workflows["scan"].Inputs["openapi"] = *profile.Inventory.Source
	if err := ValidateAuditPreparationProfile(profile); err != nil {
		t.Fatal(err)
	}
	if err := ValidateAuditTaskProfile(profile); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*ResolvedAuditProfile){
		func(p *ResolvedAuditProfile) {
			b := p.Workflows["generate-api"]
			b.MaxRunAttempts = 0
			p.Workflows["generate-api"] = b
		},
		func(p *ResolvedAuditProfile) { p.Inventory.Source.Source = AuditInputFromItemPackage },
		func(p *ResolvedAuditProfile) {
			p.Workflows["generate-api"].Inputs["existing_openapi"] = AuditWorkflowInputMapping{Source: AuditInputFromPreparation, Role: "refine-api", Name: "api"}
		},
	} {
		candidate := cloneAuditProfile(profile)
		mutate(&candidate)
		candidate.Ref.Digest, err = auditProfileDigest(Selector{ID: candidate.Ref.Name, Version: candidate.Ref.Version}, candidate)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(candidate)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeResolvedAuditProfileSnapshot(encoded); err == nil {
			t.Fatal("invalid snapshot accepted despite a recomputed digest")
		}
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	obsolete := strings.Replace(string(encoded), `"implementation":"openapi-scans@1"`, `"implementation":"openapi-scans@1","sourceInput":"source"`, 1)
	if _, err := DecodeResolvedAuditProfileSnapshot([]byte(obsolete)); err == nil {
		t.Fatal("obsolete snapshot schema accepted")
	}
}
