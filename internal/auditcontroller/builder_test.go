package auditcontroller

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestBatchedTaskPayloadBoundFitsTwoSixMiBTasksAndRejectsThird(t *testing.T) {
	selectedBytes := int64(6 << 20)
	if !fitsBatchedTaskPayload(selectedBytes, 6<<20) {
		t.Fatal("two 6 MiB task packages should fit the conservative task-set envelope")
	}
	selectedBytes += 6 << 20
	if fitsBatchedTaskPayload(selectedBytes, 6<<20) {
		t.Fatal("three 6 MiB task packages exceed the Audit task-set envelope")
	}
	if fitsBatchedTaskPayload(auditdomain.MaximumArchiveBytes, 1) {
		t.Fatal("a maximum-size single task must not be wrapped with another member")
	}
}

func TestBatchedTaskPayloadReserveCoversCanonicalZIPStructure(t *testing.T) {
	inputs := make([]auditdomain.PackageInput, auditstore.MaxCollectionItems)
	for index := range inputs {
		inputs[index] = auditdomain.PackageInput{
			ID: fmt.Sprintf("task-%03d", index), Path: fmt.Sprintf("tasks/%03d.zip", index),
			MediaType: auditdomain.PackageMediaType, Data: []byte{},
		}
	}
	payload, _, err := auditdomain.BuildPackage(
		"task-set-structural-bound", auditdomain.PackageKindTaskSet, "", inputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	archive, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
	if err != nil {
		t.Fatal(err)
	}
	manifestBytes := 0
	for _, member := range archive.File {
		if member.Name == "manifest.json" {
			manifestBytes = int(member.UncompressedSize64)
		}
	}
	if manifestBytes == 0 {
		t.Fatal("canonical task set has no manifest bytes")
	}
	structuralBytes := len(payload) - manifestBytes
	if structuralBytes > maximumTaskSetZIPStructureBytes {
		t.Fatalf("canonical task-set ZIP structure uses %d bytes, maximum %d", structuralBytes, maximumTaskSetZIPStructureBytes)
	}
}

func TestReadRoundExecutionManifestUsesExactValidatedWorklistPackage(t *testing.T) {
	taskRevision := "task-r1"
	taskRef := contracts.ArtifactRef{
		Namespace: "audit-fixture", Name: "task", Revision: &taskRevision,
	}
	manifest := auditdomain.ExecutionManifest{
		Schema: auditdomain.ExecutionManifestSchema,
		Items: []auditdomain.ExecutionItem{{
			ItemKey: "finding-one", Ordinal: 0, SubjectKey: "subject-one",
			TaskPackageID: "task-one", TaskPackageDigest: builderExact("task", taskRevision).Digest,
			TaskRef: &taskRef, Inputs: []auditdomain.ExactInput{},
		}},
	}
	encoded, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	archive, pkg, err := auditdomain.BuildPackage(
		"round-two", auditdomain.PackageKindWorklist, "",
		[]auditdomain.PackageInput{{
			ID: "execution-manifest", Path: "execution.json",
			MediaType: auditdomain.JSONMediaType, Data: encoded,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	revision := "round-r1"
	descriptor := auditstore.ExactArtifact{
		Ref: contracts.ArtifactRef{
			Namespace: "audit-fixture", Name: "round-two", Revision: &revision,
		},
		Digest: pkg.Digest,
	}
	builder := &PinnedSubmissionBuilder{artifacts: &fakeBuilderArtifactAccess{
		payload: artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: archive},
	}}
	decoded, err := builder.readRoundExecutionManifest(context.Background(), "project-one", descriptor)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Items) != 1 || decoded.Items[0].ItemKey != "finding-one" {
		t.Fatalf("decoded current Round manifest = %+v", decoded)
	}
}

func TestResolveInputsForksTrustedExecutionManifestWithoutSelfReference(t *testing.T) {
	source := builderExact("source", "source-r1")
	task := builderExact("task", "task-r1")
	execution := builderExact("execution-manifest", "execution-r1")
	binding := config.ResolvedAuditWorkflowBinding{
		Kind: config.AuditWorkflowCheck,
		Inputs: map[string]config.AuditWorkflowInputMapping{
			"source":             {Source: config.AuditInputFromAudit, Name: "source"},
			"task":               {Source: config.AuditInputFromItemPackage},
			"execution_manifest": {Source: config.AuditInputFromExecutionManifest},
		},
	}
	baseline := auditservice.BaselineSnapshot{
		Inputs: map[string]auditstore.ExactArtifact{"source": source},
	}
	manifest := auditdomain.ExecutionItem{Inputs: []auditdomain.ExactInput{{
		Name: "source", Ref: source.Ref, Digest: source.Digest,
	}}}

	runInputs, memberInputs, err := resolveInputs(binding, baseline, manifest, task, execution)
	if err != nil {
		t.Fatal(err)
	}
	if len(runInputs) != 3 || !sameExactRef(runInputs["source"].Ref, source.Ref) ||
		!sameExactRef(runInputs["task"].Ref, task.Ref) ||
		!sameExactRef(runInputs["execution_manifest"].Ref, execution.Ref) {
		t.Fatalf("Run inputs = %+v", runInputs)
	}
	if len(memberInputs) != 1 || !sameExactRef(memberInputs[0].Ref, source.Ref) {
		t.Fatalf("execution member inputs unexpectedly include the self manifest: %+v", memberInputs)
	}
}

func TestResolveRoleInputsPinsRetainedDependencyByRoundAndLogicalName(t *testing.T) {
	baselineInput := builderExact("source", "source-r1")
	retained := builderExact("inventory", "inventory-r1")
	execution := builderExact("role-manifest", "manifest-r1")
	lookup := &fakeRoleOutputLookup{link: auditstore.ArtifactLink{Artifact: retained}}
	builder := &PinnedSubmissionBuilder{outputs: lookup}
	binding := config.ResolvedAuditWorkflowBinding{
		Inputs: map[string]config.AuditWorkflowInputMapping{
			"source":   {Source: config.AuditInputFromAudit, Name: "source"},
			"manifest": {Source: config.AuditInputFromExecutionManifest},
			"context":  {Source: config.AuditInputFromRetainedOutput, Role: "discover", Name: "inventory"},
		},
		Workflow: config.ResolvedWorkflow{Inputs: map[string]config.ArtifactSlot{
			"source": {Required: true}, "manifest": {Required: true}, "context": {Required: true},
		}},
	}
	snapshot := auditstore.ReconcileSnapshot{
		Audit: auditstore.Audit{AuditID: "audit-one"},
		Round: &auditstore.Round{Ordinal: 3},
	}

	inputs, err := builder.resolveRoleInputs(
		context.Background(), snapshot, "assess", binding,
		auditservice.BaselineSnapshot{Inputs: map[string]auditstore.ExactArtifact{"source": baselineInput}},
		execution,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs) != 3 || inputs["source"].Digest != baselineInput.Digest ||
		inputs["manifest"].Digest != execution.Digest || inputs["context"].Digest != retained.Digest {
		t.Fatalf("resolved role inputs = %+v", inputs)
	}
	if lookup.auditID != "audit-one" ||
		lookup.logicalKey != auditstore.RoleOutputLogicalKey(3, "discover", "inventory") {
		t.Fatalf("retained lookup = (%q, %q)", lookup.auditID, lookup.logicalKey)
	}
}

type fakeRoleOutputLookup struct {
	link       auditstore.ArtifactLink
	auditID    string
	logicalKey string
}

type fakeBuilderArtifactAccess struct{ payload artifacts.Payload }

func (f *fakeBuilderArtifactAccess) PutImmutableProject(
	context.Context, string, contracts.ArtifactRef, artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	return auditstore.ExactArtifact{}, errors.New("unexpected immutable write")
}

func (f *fakeBuilderArtifactAccess) ResolveProjectExact(
	_ context.Context, _ string, descriptor auditstore.ExactArtifact,
) (auditstore.ExactArtifact, error) {
	descriptor.MediaType = f.payload.MediaType
	descriptor.SizeBytes = int64(len(f.payload.Data))
	return descriptor, nil
}

func (f *fakeBuilderArtifactAccess) ReadProjectExact(
	context.Context, string, auditstore.ExactArtifact,
) (artifacts.Payload, error) {
	return artifacts.Payload{
		MediaType: f.payload.MediaType,
		Data:      append([]byte(nil), f.payload.Data...),
	}, nil
}

func (f *fakeRoleOutputLookup) GetArtifactLink(
	_ context.Context, auditID, logicalKey string,
) (auditstore.ArtifactLink, error) {
	f.auditID, f.logicalKey = auditID, logicalKey
	return f.link, nil
}

func builderExact(name, revision string) auditstore.ExactArtifact {
	return auditstore.ExactArtifact{
		Ref:       contracts.ArtifactRef{Namespace: "audit-fixture", Name: name, Revision: &revision},
		Digest:    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		MediaType: "application/json", SizeBytes: 1,
	}
}
