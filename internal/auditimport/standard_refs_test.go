package auditimport

import (
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestStandardProposalReferencesRequirePinnedExistingEntries(t *testing.T) {
	pkg := top10ImportPackage(t)
	revision := "standard-r1"
	pinned := auditstandards.PinnedPackage{
		Reference: pkg.Reference(), Title: pkg.Document.Standard.Title,
		Source: pkg.Document.Standard.Source, License: pkg.Document.Standard.License,
		Retained: auditstandards.ExactPackage{
			Artifact: contracts.ArtifactRef{Namespace: "audit-example", Name: "standard", Revision: &revision},
			Digest:   pkg.Digest, MediaType: auditstandards.MediaType, SizeBytes: int64(len(pkg.Payload())),
		},
	}
	pinned.Catalog = pinned.Retained
	pinned.Catalog.Artifact.Namespace = auditstandards.CatalogNamespace
	standards := retainedStandardIndex{
		retainedStandardKey("owasp-web-top10", "2025"): {pinned: pinned, pkg: pkg},
	}
	proposal := auditdomain.FindingProposal{StandardRefs: []auditdomain.StandardReference{{
		Scheme: "owasp-web-top10", Version: "2025", RequirementID: "A01:2025",
	}}}
	if err := validateProposalStandardRefs(proposal, standards); err != nil {
		t.Fatalf("exact standard reference rejected: %v", err)
	}
	proposal.StandardRefs[0].RequirementID = "A99:2025"
	if err := validateProposalStandardRefs(proposal, standards); err == nil {
		t.Fatal("unknown standard entry was accepted")
	}
	proposal.StandardRefs[0] = auditdomain.StandardReference{
		Scheme: "owasp-web-top10", Version: "2021", RequirementID: "A01:2025",
	}
	if err := validateProposalStandardRefs(proposal, standards); err == nil {
		t.Fatal("unpinned standard version was accepted")
	}
}

func TestStandardTaskMatchesItsExactRetainedMapping(t *testing.T) {
	pkg := top10ImportPackage(t)
	revision := "standard-r1"
	ref := contracts.ArtifactRef{Namespace: "audit-example", Name: "standard", Revision: &revision}
	pinned := auditstandards.PinnedPackage{
		Reference: pkg.Reference(), Title: pkg.Document.Standard.Title,
		Source: pkg.Document.Standard.Source, License: pkg.Document.Standard.License,
		Retained: auditstandards.ExactPackage{
			Artifact: ref, Digest: pkg.Digest, MediaType: auditstandards.MediaType,
			SizeBytes: int64(len(pkg.Payload())),
		},
	}
	inventory, err := auditdomain.BuildStandardMappingInventory(pkg, auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "standard", SourceRef: ref,
		ApprovalRequirement: auditdomain.ApprovalNone,
	})
	if err != nil {
		t.Fatal(err)
	}
	task := inventory.Tasks[0].Document
	standard := retainedStandard{pinned: pinned, pkg: pkg}
	if !standardTaskMatchesPackage(task, standard) {
		t.Fatal("exact generated task did not match retained mapping")
	}
	task.Standard.EntryIDs = []string{"A02:2025"}
	if standardTaskMatchesPackage(task, standard) {
		t.Fatal("task rebound to another category matched retained mapping")
	}
	task = inventory.Tasks[0].Document
	task.Checklist.ReviewPolicy = "manual"
	if standardTaskMatchesPackage(task, standard) {
		t.Fatal("task with an invented standard review policy matched retained mapping")
	}
}

func TestStandardEvidenceContractRejectsInventedAssessmentAndKind(t *testing.T) {
	contract := auditdomain.StandardEvidenceContract{
		Assessments:   []string{"inconclusive", "refuted", "supported"},
		EvidenceKinds: []string{"observation"}, MinimumEvidence: 1, MaximumEvidence: 2,
	}
	result := auditdomain.CheckResult{
		Assessment: "supported", EvidenceIDs: []string{"ev-1"},
	}
	evidence := map[string]validatedEvidence{
		"ev-1": {value: auditdomain.Evidence{ID: "ev-1", Kind: "observation"}},
	}
	if !standardEvidenceContractAccepts(contract, result, evidence, true) {
		t.Fatal("valid standard evidence contract was rejected")
	}
	result.Assessment = "satisfied"
	if standardEvidenceContractAccepts(contract, result, evidence, true) {
		t.Fatal("assessment outside the standard contract was accepted")
	}
	result.Assessment = "supported"
	evidence["ev-1"] = validatedEvidence{value: auditdomain.Evidence{ID: "ev-1", Kind: "tool-result"}}
	if standardEvidenceContractAccepts(contract, result, evidence, true) {
		t.Fatal("evidence kind outside the standard contract was accepted")
	}
}

func top10ImportPackage(t *testing.T) auditstandards.Package {
	t.Helper()
	_, pkg, err := auditstandards.PackageDirectory(filepath.Join(
		"..", "..", "configs", "audit-standards", "owasp-web-top10-2025",
	))
	if err != nil {
		t.Fatal(err)
	}
	return *pkg
}
