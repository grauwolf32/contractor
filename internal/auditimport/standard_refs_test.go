package auditimport

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
)

func TestPinnedStandardsRequiresArrayInBaseline(t *testing.T) {
	for _, test := range []struct {
		name    string
		payload string
		valid   bool
	}{
		{name: "missing", payload: `{"schema":"contractor.audit.baseline.v1"}`},
		{name: "null", payload: `{"schema":"contractor.audit.baseline.v1","standards":null}`},
		{name: "empty", payload: `{"schema":"contractor.audit.baseline.v1","standards":[]}`, valid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			snapshot := auditstore.ReconcileSnapshot{Audit: auditstore.Audit{
				BaselineSnapshot: json.RawMessage(test.payload),
			}}
			standards, err := (&Importer{}).loadPinnedStandards(context.Background(), snapshot)
			if test.valid {
				if err != nil || standards == nil || len(standards) != 0 {
					t.Fatalf("valid empty standards = (%v, %v)", standards, err)
				}
			} else if !errors.Is(err, ErrPermanent) {
				t.Fatalf("invalid standards = %v, want ErrPermanent", err)
			}
		})
	}
}

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
	pinned.Catalog.Artifact.Namespace = artifactpolicy.AuditStandardCatalogNamespace
	standards := retainedStandardIndex{
		retainedStandardKey("owasp-web-top10", "2025"): indexRetainedStandard(pinned, pkg),
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

func TestBundledCWEClassificationMatchesRuntime(t *testing.T) {
	runtimeCatalog, err := os.ReadFile(filepath.Join(
		"..", "..", "runtime", "src", "contractor_runtime", "toolsets",
		"common", "cwe_catalog.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bundledCWECatalogJSON, runtimeCatalog) {
		t.Fatal("Go and Runtime CWE catalogs differ")
	}
	catalog, err := bundledCWECatalog()
	if err != nil {
		t.Fatal(err)
	}
	if catalog.scheme != "CWE" || catalog.version != "4.20" {
		t.Fatalf("unexpected bundled CWE identity: %s@%s", catalog.scheme, catalog.version)
	}
	proposal := auditdomain.FindingProposal{StandardRefs: []auditdomain.StandardReference{{
		Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89",
	}}}
	if err := validateProposalStandardRefs(proposal, nil); err != nil {
		t.Fatalf("Runtime-produced CWE reference rejected: %v", err)
	}
	for _, valid := range []string{"CWE-532", "CWE-639"} {
		proposal.StandardRefs[0].RequirementID = valid
		if err := validateProposalStandardRefs(proposal, nil); err != nil {
			t.Fatalf("valid CWE %s rejected: %v", valid, err)
		}
	}
	for _, invalid := range []auditdomain.StandardReference{
		{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-999999"},
		{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-596"},
		{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-534"},
		{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-1324"},
		{Scheme: "CWE", Version: "4.19", RequirementID: "CWE-89"},
	} {
		proposal.StandardRefs = []auditdomain.StandardReference{invalid}
		if err := validateProposalStandardRefs(proposal, nil); err == nil {
			t.Fatalf("invalid CWE reference accepted: %+v", invalid)
		}
	}
	proposal.StandardRefs = []auditdomain.StandardReference{{
		Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89",
	}, {
		Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89",
	}}
	if err := validateProposalStandardRefs(proposal, nil); err == nil {
		t.Fatal("duplicate CWE classification accepted")
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
	standard := indexRetainedStandard(pinned, pkg)
	if !standardTaskMatchesPackage(task, standard) {
		t.Fatal("exact generated task did not match retained mapping")
	}
	if err := validateMappedProposalStandards(task, auditdomain.FindingProposal{
		StandardRefs: []auditdomain.StandardReference{{
			Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89",
		}},
	}, retainedStandardIndex{retainedStandardKey(pkg.Reference().Scheme, pkg.Reference().Version): standard}); err == nil {
		t.Fatal("CWE classification satisfied an assigned standard entry")
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

	_, selectedPackage, err := auditstandards.PackageDirectory(
		filepath.Join(auditStandardsFixture, "owasp-asvs-5.0.0"),
	)
	if err != nil {
		t.Fatal(err)
	}
	selectedRevision := "selected-standard-r1"
	selectedRef := contracts.ArtifactRef{
		Namespace: "audit-example", Name: "selected-standard", Revision: &selectedRevision,
	}
	selectedPinned := auditstandards.PinnedPackage{
		Reference: selectedPackage.Reference(), Title: selectedPackage.Document.Standard.Title,
		Source: selectedPackage.Document.Standard.Source, License: selectedPackage.Document.Standard.License,
		Retained: auditstandards.ExactPackage{
			Artifact: selectedRef, Digest: selectedPackage.Digest, MediaType: auditstandards.MediaType,
			SizeBytes: int64(len(selectedPackage.Payload())),
		},
	}
	selectedInventory, err := auditdomain.BuildStandardMappingInventory(*selectedPackage, auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: "check", SourceInputName: "standard", SourceRef: selectedRef,
		ApprovalRequirement: auditdomain.ApprovalNone,
		StandardSelection: &auditdomain.StandardSelection{
			Scope: "one exact requirement", Levels: []string{"1"},
			EntryIDs: []string{"v5.0.0-1.2.4"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	selectedTask := selectedInventory.Tasks[0].Document
	selectedStandard := indexRetainedStandard(selectedPinned, *selectedPackage)
	if selectedTask.Checklist.Statement == selectedPackage.Document.Mappings[0].Objective ||
		!standardTaskMatchesPackage(selectedTask, selectedStandard) {
		t.Fatal("exact selected standard statement did not match its retained mapping")
	}
	selectedTask.Checklist.Statement += " model-authored suffix"
	if standardTaskMatchesPackage(selectedTask, selectedStandard) {
		t.Fatal("task with non-authoritative selected statement matched retained mapping")
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

// auditStandardsFixture holds frozen copies of the standard packages these
// tests resolve references against; they never read the operator configs/ tree.
const auditStandardsFixture = "testdata/audit-standards"

func top10ImportPackage(t *testing.T) auditstandards.Package {
	t.Helper()
	_, pkg, err := auditstandards.PackageDirectory(
		filepath.Join(auditStandardsFixture, "owasp-web-top10-2025"),
	)
	if err != nil {
		t.Fatal(err)
	}
	return *pkg
}

// pinnedStandardCollection is an import harness whose Audit pins one retained
// standard and whose result selects two proposals that reference it.
type pinnedStandardCollection struct {
	harness  importHarness
	counter  *standardReadCounter
	findings *fakeFindingRetention
	pinned   auditstandards.PinnedPackage
}

func newPinnedStandardCollection(t *testing.T) pinnedStandardCollection {
	t.Helper()
	harness := newImportHarness(t)
	profile := loadResultProfile(t)
	profileSnapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	harness.snapshot.Audit.Profile = auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest}
	harness.snapshot.Audit.ProfileSnapshot = profileSnapshot
	pkg := top10ImportPackage(t)
	revision := "standard-r1"
	pinned := auditstandards.PinnedPackage{
		Reference: pkg.Reference(), Title: pkg.Document.Standard.Title,
		Source: pkg.Document.Standard.Source, License: pkg.Document.Standard.License,
		Retained: auditstandards.ExactPackage{
			Artifact: contracts.ArtifactRef{Namespace: "audit-test", Name: "retained-standard", Revision: &revision},
			Digest:   pkg.Digest, MediaType: auditstandards.MediaType, SizeBytes: int64(len(pkg.Payload())),
		},
	}
	pinned.Catalog = pinned.Retained
	pinned.Catalog.Artifact.Namespace = artifactpolicy.AuditStandardCatalogNamespace
	pinned.Catalog.Artifact.Name = auditstandards.ArtifactName(pkg.Reference())
	if err := auditstandards.ValidatePinnedPackage(pinned); err != nil {
		t.Fatal(err)
	}
	harness.snapshot.Audit.BaselineSnapshot = standardBaseline(t, pinned)
	standardKey := refKey(pinned.Retained.Artifact)
	harness.artifacts.project[standardKey] = pkg.Payload()
	counter := &standardReadCounter{ArtifactAccess: harness.artifacts, key: standardKey}
	findings := &fakeFindingRetention{}
	selections := make([]auditdomain.ProposalSelection, 0, 2)
	for index := range 2 {
		name := fmt.Sprintf("candidate-%d", index)
		proposalRevision := "proposal-r1"
		proposal := findingintake.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-findings", Name: name, Revision: &proposalRevision},
			Digest: auditdomain.DigestBytes([]byte(name)), MediaType: "application/json", SizeBytes: int64(len(name)),
		}
		document := auditdomain.FindingProposal{StandardRefs: []auditdomain.StandardReference{{Scheme: pkg.Reference().Scheme, Version: pkg.Reference().Version, RequirementID: "A01:2025"}}}
		if index == 0 {
			document.StandardRefs = append(document.StandardRefs, auditdomain.StandardReference{
				Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89",
			})
		}
		origin := findingintake.Origin{
			RunID: *harness.execution.RunID,
			Audit: &findingintake.AuditOrigin{AuditID: harness.execution.AuditID, ExecutionID: harness.execution.ExecutionID, Role: string(harness.execution.Role)},
		}
		findings.receipts = append(findings.receipts, findingintake.Receipt{ReceiptID: name, Proposal: proposal, Document: document, Origin: origin})
		findings.resolved = append(findings.resolved, findingintake.ResolvedProposal{ReceiptID: name, Proposal: proposal, Document: document, Origin: origin})
		selections = append(selections, auditdomain.ProposalSelection{InvocationID: "worker-invocation", ClientKey: name})
	}
	harness.importer, err = New(harness.store, harness.store, harness.importer.runs, counter, findings)
	if err != nil {
		t.Fatal(err)
	}
	rebuildHarnessResult(t, &harness, selections)
	return pinnedStandardCollection{harness: harness, counter: counter, findings: findings, pinned: pinned}
}

func standardBaseline(t *testing.T, pinned ...auditstandards.PinnedPackage) json.RawMessage {
	t.Helper()
	baseline, err := json.Marshal(struct {
		Schema    string                         `json:"schema"`
		Standards []auditstandards.PinnedPackage `json:"standards"`
	}{Schema: "contractor.audit.baseline.v1", Standards: pinned})
	if err != nil {
		t.Fatal(err)
	}
	return baseline
}

// markProposalsRetained models a retry after an earlier attempt retained every
// proposal: retention then skips them and never reads the pinned standards.
func (c pinnedStandardCollection) markProposalsRetained() {
	c.findings.collection = make([]findingintake.CollectionReceipt, len(c.findings.receipts))
	for index, receipt := range c.findings.receipts {
		c.findings.collection[index] = findingintake.CollectionReceipt{
			Receipt: receipt, Retained: true, PostTerminalRetained: true,
		}
	}
}

// One collection uses the same pinned standard during receipt retention and
// validation of several result proposals. A later attempt must read it anew.
func TestCollectionLoadsPinnedStandardsOncePerAttempt(t *testing.T) {
	fixture := newPinnedStandardCollection(t)
	harness, counter, findings := fixture.harness, fixture.counter, fixture.findings
	for attempt := 1; attempt <= 2; attempt++ {
		changed, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
		if err != nil || !changed || harness.store.collected.Disposition != auditstore.CollectionAccepted {
			t.Fatalf("attempt %d: changed=%t err=%v collection=%+v", attempt, changed, err, harness.store.collected)
		}
		if counter.reads != attempt {
			t.Fatalf("after attempt %d standard reads=%d", attempt, counter.reads)
		}
		if got := len(harness.store.collected.Items[0].FindingAssociations); got != 2 {
			t.Fatalf("associations=%d", got)
		}
		if len(findings.imports) != attempt*2 || len(findings.rejections) != 0 {
			t.Fatalf("CWE proposal was not retained: imports=%d rejections=%v", len(findings.imports), findings.rejections)
		}
	}
	harness.artifacts.project[counter.key] = []byte("corrupted pinned package")
	harness.store.collected = auditstore.CollectParams{}
	changed, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
	if changed || !errors.Is(err, artifacts.ErrArtifactNotFound) || counter.reads != 3 || harness.store.collected.Disposition != "" {
		t.Fatalf("later attempt reused cached package: changed=%t err=%v reads=%d", changed, err, counter.reads)
	}
}

// When an earlier attempt retained every proposal, the first read of the
// pinned standards happens while resolving the result's proposal selections.
// That read keeps its failure class: a storage failure leaves the execution
// collecting without a receipt, and invalid pinned standards settle through
// the contract-invalid receipt. Neither is the Worker's invalid result.
func TestCollectionKeepsPinnedStandardFailureClassAfterProposalsWereRetained(t *testing.T) {
	t.Run("transient read retries without a receipt", func(t *testing.T) {
		fixture := newPinnedStandardCollection(t)
		fixture.markProposalsRetained()
		harness := fixture.harness
		transient := errors.New("artifact storage unavailable")
		fixture.counter.fail = transient
		changed, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
		if changed || !errors.Is(err, transient) || errors.Is(err, ErrPermanent) ||
			harness.store.collected.ExecutionID != "" || fixture.counter.reads != 1 ||
			len(fixture.findings.imports) != 0 || len(fixture.findings.rejections) != 0 {
			t.Fatalf("transient standard read = (%t, %v, reads=%d, collection=%+v)",
				changed, err, fixture.counter.reads, harness.store.collected)
		}
		fixture.counter.fail = nil
		changed, err = harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
		if err != nil || !changed || harness.store.collected.Disposition != auditstore.CollectionAccepted ||
			len(harness.store.collected.Items[0].FindingAssociations) != 2 || fixture.counter.reads != 2 {
			t.Fatalf("retry after transient standard read = (%t, %v, %+v)", changed, err, harness.store.collected)
		}
	})
	for name, change := range map[string]func(*pinnedStandardCollection){
		"invalid baseline": func(c *pinnedStandardCollection) {
			c.harness.snapshot.Audit.BaselineSnapshot = json.RawMessage(`{"schema":"contractor.audit.baseline.v1"}`)
		},
		"retained package differs from its pin": func(c *pinnedStandardCollection) {
			pinned := c.pinned
			pinned.Title += " (changed)"
			c.harness.snapshot.Audit.BaselineSnapshot = standardBaseline(t, pinned)
		},
	} {
		t.Run(name, func(t *testing.T) {
			fixture := newPinnedStandardCollection(t)
			fixture.markProposalsRetained()
			change(&fixture)
			harness := fixture.harness
			changed, err := harness.importer.Collect(context.Background(), harness.claim, harness.snapshot, harness.execution)
			collected := harness.store.collected
			if err != nil || !changed || collected.Disposition != auditstore.CollectionContractInvalid ||
				collected.ErrorCode == nil || *collected.ErrorCode != "collection-contract-invalid" ||
				len(collected.Items) != 1 || collected.Items[0].Retryable ||
				collected.Items[0].Coverage.Status != auditstore.CoverageBlocked {
				t.Fatalf("invalid pinned standard = (%t, %v, %+v)", changed, err, collected)
			}
		})
	}
}

// The task/standard check compares two pinned documents, so its failure is
// ErrPermanent; only the proposal's own standard references are the Worker's.
func TestResolvedProposalStandardsSeparatePinnedDataFromProposalViolations(t *testing.T) {
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
	standards := retainedStandardIndex{
		retainedStandardKey(pkg.Reference().Scheme, pkg.Reference().Version): indexRetainedStandard(pinned, pkg),
	}
	loaded := func() (retainedStandardIndex, error) { return standards, nil }
	assigned := make([]auditdomain.StandardReference, 0, len(task.Standard.EntryIDs))
	for _, entryID := range task.Standard.EntryIDs {
		assigned = append(assigned, auditdomain.StandardReference{
			Scheme: task.Standard.Scheme, Version: task.Standard.Version, RequirementID: entryID,
		})
	}
	valid := auditdomain.FindingProposal{StandardRefs: assigned}
	if err := validateResolvedProposalStandards(task, valid, loaded); err != nil {
		t.Fatalf("proposal naming its assigned entries = %v", err)
	}
	var invalid *invalidCheckResults
	omitted := auditdomain.FindingProposal{StandardRefs: []auditdomain.StandardReference{{
		Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89",
	}}}
	if err := validateResolvedProposalStandards(task, omitted, loaded); !errors.As(err, &invalid) || errors.Is(err, ErrPermanent) {
		t.Fatalf("proposal omitting its assigned entry = %v, want invalid result", err)
	}
	checklist := *task.Checklist
	checklist.ReviewPolicy = "manual"
	mismatched := task
	mismatched.Checklist = &checklist
	if err := validateResolvedProposalStandards(mismatched, valid, loaded); !errors.Is(err, ErrPermanent) || errors.As(err, &invalid) {
		t.Fatalf("pinned task differing from its retained standard = %v, want ErrPermanent", err)
	}
	transient := errors.New("artifact storage unavailable")
	if err := validateResolvedProposalStandards(task, valid, func() (retainedStandardIndex, error) {
		return nil, transient
	}); !errors.Is(err, transient) || errors.Is(err, ErrPermanent) || errors.As(err, &invalid) {
		t.Fatalf("standard read failure = %v, want it unchanged", err)
	}
	if err := validateResolvedProposalStandards(auditdomain.ItemTask{}, auditdomain.FindingProposal{}, func() (retainedStandardIndex, error) {
		t.Fatal("a proposal without standard references loaded pinned standards")
		return nil, nil
	}); err != nil {
		t.Fatalf("proposal without standard references = %v", err)
	}
}

func (c *standardReadCounter) WithClaim(claim auditstore.ControllerClaim) ArtifactAccess {
	c.ArtifactAccess = c.ArtifactAccess.WithClaim(claim)
	return c
}

type standardReadCounter struct {
	ArtifactAccess
	key   string
	reads int
	// fail, when set, is the outcome of every read of the standard.
	fail error
}

func (c *standardReadCounter) ReadProjectExact(ctx context.Context, projectID string, artifact auditstore.ExactArtifact) ([]byte, error) {
	if refKey(artifact.Ref) == c.key {
		c.reads++
		if c.fail != nil {
			return nil, c.fail
		}
	}
	return c.ArtifactAccess.ReadProjectExact(ctx, projectID, artifact)
}
