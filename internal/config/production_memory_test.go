package config

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

type memoryCatalogEntry struct {
	Retired string `json:"retired"`
	Active  string `json:"active"`
}
type memoryCatalog struct {
	SchemaVersion int                  `json:"schema_version"`
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

func TestProductionMemoryCatalogIsClosedAndRejectsRetiredSelectors(t *testing.T) {
	t.Parallel()
	catalog := repositoryMemoryCatalog(t)
	if catalog.SchemaVersion != 2 || len(catalog.Templates) != 20 || len(catalog.Workflows) != 16 || len(catalog.AuditProfiles) != 5 {
		t.Fatalf("incomplete Memory inventory: %+v", catalog)
	}
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	templates := map[string]bool{}
	for _, entry := range catalog.Templates {
		if templates[entry.Active] {
			t.Fatalf("duplicate active role %s", entry.Active)
		}
		templates[entry.Active] = true
		if _, err := snapshot.AgentTemplate(entry.Retired); err == nil {
			t.Fatalf("retired template %s: %v", entry.Retired, err)
		}
		template, err := snapshot.AgentTemplate(entry.Active)
		if err != nil {
			t.Fatal(err)
		}
		assertDigest(t, template.Ref.Digest)
		assertDigest(t, template.Instructions.Digest)
		selected := 0
		for _, toolset := range template.Toolsets {
			if toolset.Ref.ToolsetID != "memory-tools" {
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
		for _, phrase := range []string{"untrusted data", "immutable objectives", "32 KiB", "128 notes", "memory_changed", "new Run starts empty", "not result artifacts"} {
			if !strings.Contains(template.Instructions.Text, phrase) {
				t.Errorf("%s lacks %q", entry.Active, phrase)
			}
		}
	}
	workflows := map[string]bool{}
	for _, entry := range catalog.Workflows {
		if workflows[entry.Active] {
			t.Fatalf("duplicate active Workflow %s", entry.Active)
		}
		workflows[entry.Active] = true
		if _, err := snapshot.Workflow(entry.Retired); err == nil {
			t.Fatalf("retired Workflow %s: %v", entry.Retired, err)
		}
		workflow, err := snapshot.Workflow(entry.Active)
		if err != nil {
			t.Fatal(err)
		}
		for name, stage := range workflow.Stages {
			for role, binding := range stage.Agents {
				ref := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version
				if !templates[ref] {
					t.Fatalf("%s/%s/%s selects a template outside the active catalog: %s", entry.Active, name, role, ref)
				}
			}
		}
	}
	for _, entry := range catalog.AuditProfiles {
		if _, err := snapshot.AuditProfile(entry.Retired); err == nil {
			t.Fatalf("retired AuditProfile %s: %v", entry.Retired, err)
		}
		profile, err := snapshot.AuditProfile(entry.Active)
		if err != nil {
			t.Fatal(err)
		}
		for role, binding := range profile.Workflows {
			ref := binding.Workflow.Ref.Name + "@" + binding.Workflow.Ref.Version
			if !workflows[ref] {
				t.Fatalf("%s/%s selects a Workflow outside the active catalog: %s", entry.Active, role, ref)
			}
		}
	}
}

func TestRetiredWorkflowAndRebasedProfileKeepPinnedSnapshots(t *testing.T) {
	t.Parallel()
	fixture := mustLoad(t, filepath.Join("..", "..", "testdata", "configs"), MVPDescriptors())
	workflow, err := fixture.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	profile, err := fixture.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	current := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	if _, err := current.ResolveRunWorkflow(t.Context(), "artifact-copy@1", ExecutionConfigPatch{}, nil); err == nil {
		t.Fatalf("new Run silently resolved retired Workflow: %v", err)
	}
	rebased, err := current.AuditProfile("source-checklist@1")
	if err != nil || rebased.Ref.Digest == profile.Ref.Digest || rebased.Workflows["check"].WorkerCompletion == nil {
		t.Fatalf("rebased profile did not select the current completion contract: %v", err)
	}
	decodedWorkflow, err := DecodeResolvedWorkflowSnapshot(workflowJSON)
	if err != nil || !reflect.DeepEqual(workflow, decodedWorkflow) {
		t.Fatalf("pinned Workflow changed after catalog retirement: %v", err)
	}
	decodedProfile, err := DecodeResolvedAuditProfileSnapshot(profileJSON)
	if err != nil || !reflect.DeepEqual(profile, decodedProfile) {
		t.Fatalf("pinned AuditProfile changed after catalog retirement: %v", err)
	}
}
