package agentskills

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRunSkillSnapshotClonesDoNotAliasTheirSource(t *testing.T) {
	revision := "r1"
	ref := contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &revision}
	skill := RunSkillSnapshot{Name: "review", Source: &ref, Artifact: &ref}
	skillCopy := CloneRunSkills([]RunSkillSnapshot{skill})[0]
	*skillCopy.Source.Revision = "changed"
	skillCopy.Artifact.Name = "changed"
	if revision != "r1" || skill.Artifact.Name != "review" {
		t.Fatalf("a clone aliases its source: skill %+v, revision %q", skill, revision)
	}
	if CloneRunSkills(nil) != nil {
		t.Fatal("CloneRunSkills must keep nil")
	}
}

func TestRunSkillSnapshotValidatesEachLifecycleState(t *testing.T) {
	source, artifact := "source-r1", "run-r1"
	digest := "sha256:" + strings.Repeat("a", 64)
	sourceRef := contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &source}
	artifactRef := contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "review", Revision: &artifact}
	foreign := contracts.ArtifactRef{Namespace: contracts.AgentSkillNamespace, Name: "other", Revision: &source}
	selected := RunSkillSnapshot{Name: "review", Source: &sourceRef, SourceDigest: digest, SourceSize: 10}
	initialized := selected
	initialized.Artifact, initialized.PackageDigest, initialized.ExpandedBytes = &artifactRef, digest, 20
	for name, tc := range map[string]struct {
		snapshot RunSkillSnapshot
		valid    bool
	}{
		"missing marker":        {RunSkillSnapshot{Name: "review"}, true},
		"selected source":       {selected, true},
		"initialized":           {initialized, true},
		"marker with a digest":  {RunSkillSnapshot{Name: "review", SourceDigest: digest}, false},
		"foreign source":        {RunSkillSnapshot{Name: "review", Source: &foreign, SourceDigest: digest, SourceSize: 10}, false},
		"oversized source":      {RunSkillSnapshot{Name: "review", Source: &sourceRef, SourceDigest: digest, SourceSize: MaximumArchiveBytes + 1}, false},
		"result before forking": {RunSkillSnapshot{Name: "review", Source: &sourceRef, SourceDigest: digest, SourceSize: 10, ExpandedBytes: 20}, false},
		"oversized expansion": {RunSkillSnapshot{
			Name: "review", Source: &sourceRef, SourceDigest: digest, SourceSize: 10,
			Artifact: &artifactRef, PackageDigest: digest, ExpandedBytes: MaximumExpandedBytes + 1,
		}, false},
		"changed package digest": {RunSkillSnapshot{
			Name: "review", Source: &sourceRef, SourceDigest: digest, SourceSize: 10,
			Artifact: &artifactRef, PackageDigest: "sha256:" + strings.Repeat("b", 64), ExpandedBytes: 20,
		}, false},
	} {
		if err := tc.snapshot.Validate(); (err == nil) != tc.valid {
			t.Errorf("%s: Validate() = %v, want valid %t", name, err, tc.valid)
		}
	}
	if !initialized.Initialized() || selected.Initialized() {
		t.Fatal("Initialized must report a forked RunScope artifact")
	}
}
