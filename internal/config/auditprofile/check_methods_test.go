package auditprofile_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestCheckMethodAuthoringStrictRegistryAndPersistedPins(t *testing.T) {
	for _, method := range []string{"verify", "replay", "applicability", "alternative", "static-trace", "http-probe"} {
		t.Run(method, func(t *testing.T) {
			root := copyCatalog(t, "audit-profile-catalog")
			key, _ := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, method)
			text := strings.Replace(templateAuditProfileYAML(t, "routed-check"), "  workflows:", "  methods:\n    renamed-route:\n      planSchema: "+key.Schema+"\n      method: "+method+"\n      workflowRole: check\n      requiredCapabilities: []\n  workflows:", 1)
			writeAuditProfile(t, root, "routed-check", text)
			catalog, err := configload.Load(root)
			if err != nil {
				t.Fatal(err)
			}
			profile, err := catalog.AuditProfile("routed-check@1")
			if err != nil {
				t.Fatal(err)
			}
			index, err := auditprofile.CompileCheckMethods(profile)
			if err != nil {
				t.Fatal(err)
			}
			selected, ok := index.Select(key)
			if !ok || selected.Name != "renamed-route" || selected.Mapping.WorkflowRole != "check" || selected.Mapping.Phase != key.Phase {
				t.Fatalf("selected %+v,%t", selected, ok)
			}
			withoutPhase := profile.Clone()
			mapping := withoutPhase.Methods["renamed-route"]
			mapping.Phase = ""
			withoutPhase.Methods["renamed-route"] = mapping
			withoutPhaseDigest, err := auditprofile.Digest(withoutPhase)
			if err != nil || withoutPhaseDigest == profile.Ref.Digest {
				t.Fatal("resolved phase is absent from the profile digest", err)
			}
			encoded, _ := json.Marshal(profile)
			if _, err = auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err != nil {
				t.Fatal(err)
			}
			for _, edit := range []func(*auditprofile.ResolvedAuditProfile){
				func(p *auditprofile.ResolvedAuditProfile) {
					v := p.Methods["renamed-route"]
					v.Method = "operator-text"
					p.Methods["renamed-route"] = v
				},
				func(p *auditprofile.ResolvedAuditProfile) {
					v := p.Methods["renamed-route"]
					v.Phase = "guessed-phase"
					p.Methods["renamed-route"] = v
				},
				func(p *auditprofile.ResolvedAuditProfile) {
					v := p.Methods["renamed-route"]
					v.Phase = ""
					p.Methods["renamed-route"] = v
				},
				func(p *auditprofile.ResolvedAuditProfile) { p.Methods["duplicate"] = p.Methods["renamed-route"] },
				func(p *auditprofile.ResolvedAuditProfile) {
					v := p.Methods["renamed-route"]
					v.WorkflowRole = "missing"
					p.Methods["renamed-route"] = v
				},
				func(p *auditprofile.ResolvedAuditProfile) {
					v := p.Methods["renamed-route"]
					v.RequiredCapabilities = nil
					p.Methods["renamed-route"] = v
				},
			} {
				bad := profile.Clone()
				edit(&bad)
				redigest(t, &bad)
				data, _ := json.Marshal(bad)
				if _, err = auditprofile.DecodeResolvedAuditProfileSnapshot(data); err == nil {
					t.Fatal("invalid persisted method accepted")
				}
			}
		})
	}
}

func TestSharedCheckMethodAuthoringDerivesPhase(t *testing.T) {
	var fixture struct {
		AuthoredMappingCases []struct {
			Name          string          `json:"name"`
			Valid         bool            `json:"valid"`
			Mapping       json.RawMessage `json:"mapping"`
			ResolvedPhase string          `json:"resolvedPhase"`
		} `json:"authoredMappingCases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-check-plans", "accepted-plan-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.AuthoredMappingCases) == 0 {
		t.Fatal("shared authored mappings are empty")
	}
	for _, test := range fixture.AuthoredMappingCases {
		t.Run(test.Name, func(t *testing.T) {
			root := copyCatalog(t, "audit-profile-catalog")
			var mapping bytes.Buffer
			if err := json.Compact(&mapping, test.Mapping); err != nil {
				t.Fatal(err)
			}
			methods := "  methods:\n    renamed-route: " + mapping.String() + "\n  workflows:"
			manifest := strings.Replace(templateAuditProfileYAML(t, "derived-phase"), "  workflows:", methods, 1)
			writeAuditProfile(t, root, "derived-phase", manifest)
			catalog, err := configload.Load(root)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%t want %t: %v", err == nil, test.Valid, err)
			}
			if err != nil {
				return
			}
			profile, err := catalog.AuditProfile("derived-phase@1")
			if err != nil {
				t.Fatal(err)
			}
			if profile.Methods["renamed-route"].Phase != test.ResolvedPhase {
				t.Fatalf("resolved phase=%q want %q", profile.Methods["renamed-route"].Phase, test.ResolvedPhase)
			}
			encoded, err := json.Marshal(profile)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCheckMethodCapabilitiesUseExactWorkerAndClone(t *testing.T) {
	root := copyCatalog(t, "audit-profile-catalog")
	manifest := strings.Replace(templateAuditProfileYAML(t, "capability-route"), "  workflows:", `  methods:
    exact-reader:
      planSchema: contractor.audit.finding-check-plan.v1
      method: verify
      workflowRole: check
      requiredCapabilities:
        - stage: trace
          agent: analyst
          ref: {toolsetId: filesystem, version: "1"}
          tools: [read_file, ls]
  workflows:`, 1)
	writeAuditProfile(t, root, "capability-route", manifest)
	profile := loadProfile(t, root, "capability-route@1")
	mapping := profile.Methods["exact-reader"]
	if strings.Join(mapping.RequiredCapabilities[0].Tools, ",") != "ls,read_file" {
		t.Fatal("authored tools were not normalized")
	}
	index, err := auditprofile.CompileCheckMethods(profile)
	if err != nil {
		t.Fatal(err)
	}
	key, _ := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, "verify")
	selected, ok := index.Select(key)
	if !ok {
		t.Fatal("method absent")
	}
	selected.Mapping.RequiredCapabilities[0].Tools[0] = "missing"
	again, _ := index.Select(key)
	if again.Mapping.RequiredCapabilities[0].Tools[0] != "ls" {
		t.Fatal("index aliases selected capabilities")
	}
	cloned := profile.Clone()
	cloned.Methods["exact-reader"].RequiredCapabilities[0].Tools[0] = "changed"
	if mapping.RequiredCapabilities[0].Tools[0] != "ls" {
		t.Fatal("profile clone aliases capabilities")
	}
	for _, mutate := range []func(*contracts.CheckCapabilityRequirement){
		func(value *contracts.CheckCapabilityRequirement) { value.Stage = "other" },
		func(value *contracts.CheckCapabilityRequirement) { value.Agent = "other" },
		func(value *contracts.CheckCapabilityRequirement) { value.Tools = []string{"write_file"} },
		func(value *contracts.CheckCapabilityRequirement) { value.Ref.Version = "2" },
	} {
		bad := profile.Clone()
		candidate := bad.Methods["exact-reader"]
		mutate(&candidate.RequiredCapabilities[0])
		bad.Methods["exact-reader"] = candidate
		if _, err := auditprofile.CompileCheckMethods(bad); err == nil {
			t.Fatal("capability escaped its exact Worker selection")
		}
	}
}

func TestSharedCheckMethodMappingNameText(t *testing.T) {
	var fixture struct {
		MappingNameCases []struct {
			Name, Value string
			Valid       bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-check-plans", "accepted-plan-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range fixture.MappingNameCases {
		t.Run(test.Name, func(t *testing.T) {
			root := copyCatalog(t, "audit-profile-catalog")
			name, _ := json.Marshal(test.Value)
			methods := "  methods:\n    " + string(name) + ":\n      planSchema: contractor.audit.finding-check-plan.v1\n      method: verify\n      workflowRole: check\n      requiredCapabilities: []\n  workflows:"
			manifest := strings.Replace(templateAuditProfileYAML(t, "mapping-text"), "  workflows:", methods, 1)
			writeAuditProfile(t, root, "mapping-text", manifest)
			catalog, err := configload.Load(root)
			if (err == nil) != test.Valid {
				t.Fatalf("valid=%t want%t: %v", err == nil, test.Valid, err)
			}
			if err != nil {
				return
			}
			profile, err := catalog.AuditProfile("mapping-text@1")
			if err != nil {
				t.Fatal(err)
			}
			encoded, _ := json.Marshal(profile)
			if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(encoded); err != nil {
				t.Fatal(err)
			}
		})
	}
}
