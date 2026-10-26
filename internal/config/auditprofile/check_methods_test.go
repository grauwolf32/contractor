package auditprofile_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestCheckMethodAuthoringStrictRegistryAndPersistedPins(t *testing.T) {
	for _, method := range []string{"verify", "replay", "applicability", "alternative", "static-trace", "http-probe"} {
		t.Run(method, func(t *testing.T) {
			root := copyCatalog(t, "audit-profile-catalog")
			key, _ := contracts.LookupCheckPlanMethod(contracts.FindingCheckPlanSchema, method)
			text := strings.Replace(templateAuditProfileYAML(t, "routed-check"), "  workflows:", "  methods:\n    renamed-route:\n      planSchema: "+key.Schema+"\n      method: "+method+"\n      phase: "+key.Phase+"\n      workflowRole: check\n      requiredCapabilities: []\n  workflows:", 1)
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
			if err != nil || !index.Enabled() {
				t.Fatal(err)
			}
			selected, ok := index.Select(key)
			if !ok || selected.Name != "renamed-route" || selected.Mapping.WorkflowRole != "check" {
				t.Fatalf("selected %+v,%t", selected, ok)
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
