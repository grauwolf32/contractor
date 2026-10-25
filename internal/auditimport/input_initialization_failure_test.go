package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestImporterSettlesFailedInputInitializationWithoutRetry(t *testing.T) {
	for _, scan := range []bool{false, true} {
		t.Run(map[bool]string{false: "ordinary check", true: "before scan recovery"}[scan], func(t *testing.T) {
			harness := newImportHarness(t)
			if scan {
				rewriteInputFailureTask(t, &harness, true)
			}
			runs := failedInputRunReader(&harness)
			changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
			collected := harness.store.collected
			if err != nil || !changed || runs.reads != 1 || collected.Disposition != auditstore.CollectionContractInvalid ||
				collected.ErrorCode == nil || *collected.ErrorCode != runstore.AuditInputInitializationFailedReason ||
				len(collected.Items) != 1 || collected.Items[0].Retryable || collected.SourceOutput != nil ||
				len(collected.Retained) != 0 || collected.Items[0].Coverage.Status != auditstore.CoverageBlocked ||
				collected.Items[0].FinalDisposition != auditstore.FinalInvalidResult {
				t.Fatalf("input failure collection = (%t, %v, %d reads, %+v)", changed, err, runs.reads, collected)
			}
		})
	}
}

func TestImporterSettlesRejectedSubmissionWithoutChildOrRetry(t *testing.T) {
	for _, scan := range []bool{false, true} {
		t.Run(map[bool]string{false: "ordinary check", true: "scan check"}[scan], func(t *testing.T) {
			harness := newImportHarness(t)
			if scan {
				rewriteInputFailureTask(t, &harness, true)
			}
			runs := failedInputRunReader(&harness)
			outcome := auditstore.TerminalSubmissionFailed
			harness.execution.TerminalOutcome = &outcome
			harness.execution.RunID = nil
			changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
			collected := harness.store.collected
			if err != nil || !changed || runs.reads != 0 || collected.Disposition != auditstore.CollectionContractInvalid ||
				collected.ErrorCode == nil || *collected.ErrorCode != "collection-contract-invalid" ||
				len(collected.Items) != 1 || collected.Items[0].Retryable || collected.SourceOutput != nil ||
				len(collected.Retained) != 0 || collected.Items[0].FinalDisposition != auditstore.FinalInvalidResult {
				t.Fatalf("submission rejection = (%t, %v, %d reads, %+v)", changed, err, runs.reads, collected)
			}
		})
	}
}

func TestImporterFailedInputReadOutageDoesNotCommitReceipt(t *testing.T) {
	harness := newImportHarness(t)

	runs := failedInputRunReader(&harness)
	runs.err = errors.New("Run storage unavailable")
	changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
	if changed || !errors.Is(err, runs.err) || runs.reads != 1 || harness.store.collected.ExecutionID != "" {
		t.Fatalf("read outage = (%t, %v, %d reads, %+v)", changed, err, runs.reads, harness.store.collected)
	}
}

func TestImporterInputFailureRequiresExactChildOwnership(t *testing.T) {
	for name, mutate := range map[string]func(*runstore.WorkflowRun){
		"Run":         func(run *runstore.WorkflowRun) { run.RunID = "foreign" },
		"owner":       func(run *runstore.WorkflowRun) { run.OwnerID = "foreign" },
		"Project":     func(run *runstore.WorkflowRun) { run.ProjectID = stringPointer("foreign") },
		"publication": func(run *runstore.WorkflowRun) { run.PublicationMode = runstore.PublicationOrdinary },
		"execution":   func(run *runstore.WorkflowRun) { run.AuditExecutionID = stringPointer("foreign") },
		"submission":  func(run *runstore.WorkflowRun) { run.AuditSubmissionKey = stringPointer("foreign") },
		"state":       func(run *runstore.WorkflowRun) { run.State = runstore.RunPending },
	} {
		t.Run(name, func(t *testing.T) {
			harness := newImportHarness(t)

			runs := failedInputRunReader(&harness)
			mutate(&runs.run)
			changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
			if err != nil || !changed || runs.reads != 1 || harness.store.collected.Disposition != auditstore.CollectionContractInvalid ||
				harness.store.collected.ErrorCode == nil || *harness.store.collected.ErrorCode != "collection-contract-invalid" ||
				harness.store.collected.Items[0].Retryable {
				t.Fatalf("foreign child collection = (%t, %v, %+v)", changed, err, harness.store.collected)
			}
		})
	}
}

func TestImporterUnrelatedFailureKeepsRetryBehavior(t *testing.T) {
	harness := newImportHarness(t)
	runs := failedInputRunReader(&harness)
	runs.run.StateReason.Code = "worker-failed"
	changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
	if err != nil || !changed || runs.reads != 1 || harness.store.collected.Disposition != auditstore.CollectionExecutionFailed || !harness.store.collected.Items[0].Retryable {
		t.Fatalf("ordinary failure = (%t, %v, %d reads, %+v)", changed, err, runs.reads, harness.store.collected)
	}
}

func TestImporterCancelledCheckDoesNotReadInitializationFailure(t *testing.T) {
	harness := newImportHarness(t)
	runs := failedInputRunReader(&harness)
	outcome := auditstore.TerminalCancelled
	harness.execution.TerminalOutcome = &outcome
	changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
	if err != nil || !changed || runs.reads != 0 || harness.store.collected.Disposition != auditstore.CollectionExecutionCancelled {
		t.Fatalf("cancelled check = (%t, %v, %d reads, %+v)", changed, err, runs.reads, harness.store.collected)
	}
}

func TestImporterSettlesFailedAssessmentInputWithoutRoleRetry(t *testing.T) {
	for _, outcome := range []auditstore.TerminalOutcome{auditstore.TerminalFailed, auditstore.TerminalSubmissionFailed} {
		for _, declared := range []bool{false, true} {
			t.Run(string(outcome)+"/"+map[bool]string{false: "ordinary assessment", true: "round results"}[declared], func(t *testing.T) {
				harness := newImportHarness(t)
				profile := inputFailureAssessmentProfile(t, declared)
				data, err := json.Marshal(profile)
				if err != nil {
					t.Fatal(err)
				}
				harness.snapshot.Audit.ProfileSnapshot = data
				harness.snapshot.Audit.Profile = auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest}
				harness.execution.Role = auditstore.ExecutionAssessment
				harness.execution.WorkflowRole = "review"
				attempt := 1
				harness.execution.RoleAttempt = &attempt
				harness.store.members = nil
				harness.snapshot.Round = &auditstore.Round{RoundID: *harness.execution.RoundID, Ordinal: 1}
				runs := failedInputRunReader(&harness)
				harness.execution.TerminalOutcome = &outcome
				if outcome == auditstore.TerminalSubmissionFailed {
					harness.execution.RunID = nil
				}
				changed, err := harness.importer.Collect(t.Context(), harness.claim, harness.snapshot, harness.execution)
				wantDisposition := auditstore.CollectionExecutionFailed
				wantReads := 0
				if declared {
					wantDisposition = auditstore.CollectionContractInvalid
					if outcome == auditstore.TerminalFailed {
						wantReads = 1
					}
				}
				if err != nil || !changed || runs.reads != wantReads || harness.store.collected.Disposition != wantDisposition || len(harness.store.collected.Items) != 0 {
					t.Fatalf("assessment failure = (%t,%v,%d reads,%+v)", changed, err, runs.reads, harness.store.collected)
				}
				if declared && (harness.store.collected.ErrorCode == nil || *harness.store.collected.ErrorCode != runstore.AuditInputInitializationFailedReason) {
					t.Fatalf("lost input failure code: %+v", harness.store.collected)
				}
			})
		}
	}
}

type inputFailureRunReader struct {
	run   runstore.WorkflowRun
	err   error
	reads int
}

func (reader *inputFailureRunReader) GetRun(context.Context, string) (runstore.WorkflowRun, error) {
	reader.reads++
	return reader.run, reader.err
}

func failedInputRunReader(harness *importHarness) *inputFailureRunReader {
	outcome := auditstore.TerminalFailed
	harness.execution.TerminalOutcome = &outcome
	harness.execution.SubmissionKey = "submission-1"
	run := harness.importer.runs.(*fakeImportRuns).run
	run.OwnerID = harness.snapshot.Audit.OwnerID
	run.State = runstore.RunFailed
	run.StateReason = runstore.Reason{Code: runstore.AuditInputInitializationFailedReason}
	run.AuditSubmissionKey = stringPointer(harness.execution.SubmissionKey)
	reader := &inputFailureRunReader{run: run}
	harness.importer.runs = reader
	return reader
}

func rewriteInputFailureTask(t *testing.T, harness *importHarness, scan bool) {
	t.Helper()
	member := &harness.store.members[0]
	pkg, err := auditdomain.ValidatePackage(harness.artifacts.project[refKey(member.Task.Ref)])
	if err != nil {
		t.Fatal(err)
	}
	document, _ := pkg.MemberByID("task-document")
	task, err := auditdomain.DecodeItemTask(document.Data())
	if err != nil {
		t.Fatal(err)
	}
	descriptor := harness.artifacts.runDescriptor.Clone()
	descriptor.Ref.Namespace = auditdomain.ArtifactNamespace(harness.snapshot.Audit.AuditID)

	if scan {
		task.Kind = "web-scan"
		task.Checklist = nil
		task.Scan = &auditdomain.ScanTask{Scanner: "nuclei", Operation: "target", Settings: auditdomain.ExactInput{Name: "settings", Ref: descriptor.Ref, Digest: descriptor.Digest}, PreparationDigest: auditdomain.DigestBytes(nil), Runnable: true, TargetURL: "https://target.test/", TestParameters: []string{}, Gaps: []string{"url_template_scan_only"}}
	}
	data, err := auditdomain.EncodeItemTask(task)
	if err != nil {
		t.Fatal(err)
	}
	payload, validated, err := auditdomain.BuildPackage(pkg.Manifest.PackageID, auditdomain.PackageKindTask, "", []auditdomain.PackageInput{{ID: "task-document", Path: "task.json", MediaType: auditdomain.JSONMediaType, Data: data}})
	if err != nil {
		t.Fatal(err)
	}
	member.Task.Digest = validated.Digest
	member.Task.SizeBytes = int64(len(payload))
	harness.artifacts.project[refKey(member.Task.Ref)] = payload
	harness.store.selectedItems[0].Task = member.Task
	harness.store.selectedItems[0].Kind = task.Kind
}

func inputFailureAssessmentProfile(t *testing.T, declared bool) auditprofile.ResolvedAuditProfile {
	t.Helper()
	profile := loadResultProfile(t)
	binding := profile.Workflows["check"]
	binding.Kind = auditprofile.AuditWorkflowAssessment
	binding.Workflow = binding.Workflow.Clone()
	binding.Workflow.Inputs["context"] = resolved.ArtifactSlot{Required: true, MediaTypes: []string{auditdomain.JSONMediaType}}
	binding.Inputs = map[string]auditprofile.AuditWorkflowInputMapping{"task": {Source: auditprofile.AuditInputFromItemPackage}, "context": {Source: auditprofile.AuditInputFromAudit, Name: "checklist"}}
	if declared {
		binding.Inputs["context"] = auditprofile.AuditWorkflowInputMapping{Source: auditprofile.AuditInputFromRoundResults}
	}
	profile.Workflows["review"] = binding
	digest, err := auditprofile.Digest(profile)
	if err != nil {
		t.Fatal(err)
	}
	profile.Ref.Digest = digest
	data, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := auditprofile.DecodeResolvedAuditProfileSnapshot(data); err != nil {
		t.Fatal(err)
	}
	return profile
}
