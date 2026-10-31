package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedSitemapPolicyTemplateDigest(t *testing.T) {
	var fixture struct {
		Template contracts.ResolvedAgentTemplate `json:"template"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "sitemap-access-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if err := fixture.Template.Validate(); err != nil {
		t.Fatal(err)
	}
	got, err := agentTemplateDigest(resolved.Selector{ID: fixture.Template.Ref.TemplateID, Version: fixture.Template.Ref.Version}, fixture.Template)
	if err != nil || got != fixture.Template.Ref.Digest {
		t.Fatalf("sitemap digest = %s, want %s: %v", got, fixture.Template.Ref.Digest, err)
	}
}

func TestSitemapPolicyAuthoringPinsAndClonesAuthority(t *testing.T) {
	root := copyCoreFixture(t)
	appendFile(t, templatePath(root), sitemapPolicyYAML)
	snapshot := mustLoad(t, root)
	first, err := snapshot.agentTemplateForTest("artifact_builder@1")
	if err != nil || first.Sitemap == nil || !first.Sitemap.CollectObservations {
		t.Fatalf("pinned sitemap policy: %+v %v", first.Sitemap, err)
	}
	digest := first.Ref.Digest
	*first.Sitemap.Deployment = "changed"
	first.Sitemap.SnapshotInputs[0] = "changed"
	again, err := snapshot.agentTemplateForTest("artifact_builder@1")
	if err != nil || *again.Sitemap.Deployment != "prod" || len(again.Sitemap.SnapshotInputs) != 1 || again.Sitemap.SnapshotInputs[0] != "api" {
		t.Fatal("sitemap policy shares mutable template state")
	}
	replaceFile(t, templatePath(root), "collectObservations: true", "collectObservations: false")
	changed, err := mustLoad(t, root).agentTemplateForTest("artifact_builder@1")
	if err != nil || changed.Ref.Digest == digest {
		t.Fatal("collection authority is outside the template digest")
	}
}

func TestSitemapPolicyRejectsInvalidAuthoring(t *testing.T) {
	for name, block := range map[string]string{
		"null":               "  sitemap: null\n",
		"missing collection": strings.Replace(sitemapPolicyYAML, "    collectObservations: true\n", "", 1),
		"wrong version":      strings.Replace(sitemapPolicyYAML, "sitemap-access@1", "sitemap-access@2", 1),
		"obsolete mode":      sitemapPolicyYAML + "    sourceMode: blackbox\n",
		"obsolete inputs":    sitemapPolicyYAML + "    openapiInputs: [api]\n",
		"unknown field":      sitemapPolicyYAML + "    ownerId: forged\n",
		"duplicate field":    sitemapPolicyYAML + "    collectObservations: false\n",
	} {
		t.Run(name, func(t *testing.T) {
			root := copyCoreFixture(t)
			appendFile(t, templatePath(root), block)
			if _, err := LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed")); err == nil {
				t.Fatal("accepted invalid sitemap policy")
			}
		})
	}
}

const sitemapPolicyYAML = `  sitemap:
    version: sitemap-access@1
    deployment: prod
    snapshotInputs: [api]
    collectObservations: true
`
