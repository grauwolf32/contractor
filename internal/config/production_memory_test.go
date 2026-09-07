package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type memoryCatalogEntry struct {
	Legacy string `json:"legacy"`
	Active string `json:"active"`
}
type memoryCatalog struct {
	Templates     []memoryCatalogEntry `json:"templates"`
	Workflows     []memoryCatalogEntry `json:"workflows"`
	AuditProfiles []memoryCatalogEntry `json:"audit_profiles"`
}

func repositoryMemoryCatalog(t *testing.T) memoryCatalog {
	t.Helper()
	var catalog memoryCatalog
	if err := json.Unmarshal(readFile(t, filepath.Join(repositoryConfigRoot, "memory-catalog.json")), &catalog); err != nil {
		t.Fatal(err)
	}
	return catalog
}

func TestProductionMemoryCatalogPreservesDomainContracts(t *testing.T) {
	t.Parallel()
	catalog := repositoryMemoryCatalog(t)
	if len(catalog.Templates) != 20 || len(catalog.Workflows) != 16 || len(catalog.AuditProfiles) != 5 {
		t.Fatalf("incomplete Memory inventory: %+v", catalog)
	}
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	templates := map[string]string{}
	for _, entry := range catalog.Templates {
		if templates[entry.Legacy] != "" {
			t.Fatalf("duplicate legacy role %s", entry.Legacy)
		}
		templates[entry.Legacy] = entry.Active
		before, err := snapshot.AgentTemplate(entry.Legacy)
		if err != nil {
			t.Fatal(err)
		}
		after, err := snapshot.AgentTemplate(entry.Active)
		if err != nil {
			t.Fatal(err)
		}
		if before.Ref.Version == after.Ref.Version || before.Ref.Digest == after.Ref.Digest {
			t.Fatalf("%s did not advance identity/digest", entry.Active)
		}
		domain := []contracts.ToolsetSelection{}
		selected := 0
		for _, toolset := range before.Toolsets {
			if toolset.Ref.ToolsetID == "memory-tools" {
				t.Fatalf("legacy %s gained Memory", entry.Legacy)
			}
		}
		for _, toolset := range after.Toolsets {
			if toolset.Ref.ToolsetID != "memory-tools" {
				domain = append(domain, toolset)
				continue
			}
			selected++
			got := append([]string(nil), toolset.Tools...)
			sort.Strings(got)
			want := []string{"append_memory", "list_memories", "list_memory_tags", "read_memory", "search_memory", "write_memory"}
			if toolset.Ref.Version != "1" || !reflect.DeepEqual(got, want) {
				t.Fatalf("%s Memory selection = %+v", entry.Active, toolset)
			}
		}
		if selected != 1 {
			t.Fatalf("%s selects Memory %d times", entry.Active, selected)
		}
		if !strings.HasPrefix(after.Instructions.Text, strings.TrimSpace(before.Instructions.Text)) {
			t.Fatalf("%s lost original domain instructions", entry.Active)
		}
		for _, phrase := range []string{"untrusted data", "immutable objectives", "32 KiB", "128 notes", "memory_changed", "new Run starts empty", "not result artifacts"} {
			if !strings.Contains(after.Instructions.Text, phrase) {
				t.Errorf("%s lacks %q", entry.Active, phrase)
			}
		}
		after.Toolsets = domain
		after.Ref = before.Ref
		after.Instructions = before.Instructions
		if !reflect.DeepEqual(before, after) {
			t.Fatalf("%s changed domain capabilities, completion or policy", entry.Active)
		}
	}
	workflows := map[string]string{}
	for _, entry := range catalog.Workflows {
		workflows[entry.Legacy] = entry.Active
		before, err := snapshot.Workflow(entry.Legacy)
		if err != nil {
			t.Fatal(err)
		}
		after, err := snapshot.Workflow(entry.Active)
		if err != nil {
			t.Fatal(err)
		}
		if before.Ref.Version == after.Ref.Version {
			t.Fatalf("Workflow %s retained its old version", entry.Active)
		}
		for name, stage := range after.Stages {
			old := before.Stages[name]
			for role, binding := range stage.Agents {
				prior := old.Agents[role]
				oldRef := prior.Template.Ref.TemplateID + "@" + prior.Template.Ref.Version
				newRef := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version
				if templates[oldRef] != newRef {
					t.Fatalf("%s/%s/%s selects %s, want %s", entry.Active, name, role, newRef, templates[oldRef])
				}
				binding.Template = prior.Template
				stage.Agents[role] = binding
			}
			after.Stages[name] = stage
		}
		after.Ref = before.Ref
		if !reflect.DeepEqual(before, after) {
			t.Fatalf("%s changed workflow behavior beyond versioned templates", entry.Active)
		}
	}
	for _, entry := range catalog.AuditProfiles {
		before, err := snapshot.AuditProfile(entry.Legacy)
		if err != nil {
			t.Fatal(err)
		}
		after, err := snapshot.AuditProfile(entry.Active)
		if err != nil {
			t.Fatal(err)
		}
		for role, binding := range after.Workflows {
			prior := before.Workflows[role]
			if binding.Workflow.Ref.Name+"@"+binding.Workflow.Ref.Version != workflows[prior.Workflow.Ref.Name+"@"+prior.Workflow.Ref.Version] {
				t.Fatalf("%s/%s did not select its Memory Workflow", entry.Active, role)
			}
			binding.Workflow = prior.Workflow
			after.Workflows[role] = binding
		}
		after.Ref = before.Ref
		if !reflect.DeepEqual(before, after) {
			t.Fatalf("%s changed Audit obligations", entry.Active)
		}
	}
}
