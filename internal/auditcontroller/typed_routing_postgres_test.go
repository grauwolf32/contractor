//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func typedRoutingControllerConfig(t *testing.T, customize ...func(map[string]string)) *config.Snapshot {
	t.Helper()
	configure := func(files map[string]string) {
		profile := strings.Replace(files["audit-profiles/checklist.yaml"], "maxRounds: 1", "maxRounds: 3", 1)
		profile = strings.Replace(profile, "maxEvidenceBytes: 1048576", "maxEvidenceBytes: 8388608", 1)
		methods := "  methods:\n"
		for _, entry := range []struct{ method, role string }{{"verify", "blue"}, {"replay", "red"}, {"applicability", "green"}} {
			methods += fmt.Sprintf("    %s-route:\n      planSchema: contractor.audit.finding-check-plan.v1\n      method: %s\n      workflowRole: %s\n      requiredCapabilities: []\n", entry.role, entry.method, entry.role)
		}
		profile = strings.Replace(profile, "  workflows:\n", methods+"  workflows:\n", 1)
		roles := ""
		for _, role := range []string{"blue", "red", "green"} {
			roles += fmt.Sprintf("    %s:\n      kind: check\n      ref: audit-check@1\n      inputs:\n        task: {source: item-package}\n        proposal: {source: retained-dependency, name: proposal}\n        previous_result: {source: retained-dependency, name: source-result}\n      parameters: {}\n      outputs: {result: result}\n", role)
		}
		profile = strings.Replace(profile, "  execution:\n", roles+"  execution:\n", 1)
		files["audit-profiles/checklist.yaml"] = profile
		files["workflows/check.yaml"] = strings.Replace(files["workflows/check.yaml"], "    task: {required: true, mediaTypes: [application/zip]}", "    task: {required: true, mediaTypes: [application/zip]}\n    proposal: {required: false, mediaTypes: [application/json]}\n    previous_result: {required: false, mediaTypes: [application/zip]}", 1)
	}
	return loadControllerConfigWithFindings(t, 1, 8, true, append([]func(map[string]string){configure}, customize...)...)
}

type lostTypedAcceptanceReply struct {
	Store
	lost    bool
	failure error
}

func (s *lostTypedAcceptanceReply) AcceptNextRound(ctx context.Context, p auditstore.AcceptRoundParams) (auditstore.Round, bool, error) {
	round, inserted, err := s.Store.AcceptNextRound(ctx, p)
	if err == nil && inserted && p.PreparedRoot != nil && !s.lost {
		s.lost = true
		return auditstore.Round{}, false, s.failure
	}
	return round, inserted, err
}

func TestPostgresTypedRoutingUsesActualAdmissionAfterPauseRestartAndProducerDeletion(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 180*time.Second)
	defer cancel()
	h := newPostgresControllerHarnessWithConfig(t, ctx, 0, 1, typedRoutingControllerConfig(t))
	intake, err := findingintake.New(h.pool)
	if err != nil {
		t.Fatal(err)
	}
	c := h.controllerWithCollector(t, intake)
	c.roundBuilder = h.service
	c.options.NewID = randomid.New
	for range 2 {
		preparationStep(t, ctx, c)
	}
	executions, err := h.audits.ListExecutions(ctx, h.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("seed execution: %+v %v", executions, err)
	}
	producer := executions[0]
	if producer.RoundID == nil {
		t.Fatal("seed check has no Round")
	}
	// An ordinary finding may propose no checks. Its indexed marker must
	// advance the cursor before the following receipt rather than stall the
	// durable admission intent at an empty first page.
	grant := childFindingGrant(t, ctx, h, *producer.RunID)
	submitChildFindingProposal(t, ctx, intake, grant, "no-follow-up-checks", nil)
	receipt := submitChildFindingProposal(t, ctx, intake, grant, "typed-plans", nil,
		auditdomain.ProposedCheck{Objective: "Verify exact accepted result.", Method: "verify"},
		auditdomain.ProposedCheck{Objective: "Replay the exact receipt.", Method: "replay"},
		auditdomain.ProposedCheck{Objective: "Establish applicability from exact materials.", Method: "applicability"},
		auditdomain.ProposedCheck{Objective: "This unregistered method has no route.", Method: "unregistered"})
	finishTypedRoutingCheck(t, ctx, h, producer)
	var before auditstore.AdmissionProgress
	for step := 0; step < 20; step++ {
		preparationStep(t, ctx, c)
		before = readTypedAdmissionProgress(t, ctx, h)
		if before.SelectedCount == 1 {
			break
		}
	}
	if before.SelectedCount != 1 || before.Root != nil {
		t.Fatalf("bounded first selection: %+v", before)
	}
	current, err := h.audits.Get(ctx, h.started.Audit.OwnerID, h.started.Audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	paused, err := h.service.Pause(ctx, auditservice.MutationParams{OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "pause-typed", RequestDigest: postgresDigest("pause-typed")})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.RunOnce(ctx); err != nil {
		t.Fatal(err)
	}
	if after := readTypedAdmissionProgress(t, ctx, h); after.Revision != before.Revision || after.SelectedCount != 1 {
		t.Fatal("pause advanced typed membership")
	}
	if err := h.runs.MarkStageAllocationReleased(ctx, *producer.RunID+"-allocation"); err != nil {
		t.Fatal(err)
	}
	if err := rundeletion.New(h.pool).DeleteReleasedTerminalRun(ctx, paused.Audit.OwnerID, *producer.RunID); err != nil {
		t.Fatal(err)
	}
	current, err = h.audits.Get(ctx, current.OwnerID, current.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.Resume(ctx, auditservice.MutationParams{OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "resume-typed", RequestDigest: postgresDigest("resume-typed")}); err != nil {
		t.Fatal(err)
	}
	c = h.controllerWithCollector(t, intake)
	c.roundBuilder = h.service
	c.options.NewID = randomid.New
	lost := &lostTypedAcceptanceReply{Store: c.store, failure: errors.New("accepted Round reply was lost")}
	c.store = lost
	completed := map[string]bool{}
	methods := map[string]string{}
	for step := 0; step < 70; step++ {
		_, err := c.RunOnce(ctx)
		if err != nil && !errors.Is(err, lost.failure) {
			t.Fatalf("typed step%d: %v", step, err)
		}
		executions, err = h.audits.ListExecutions(ctx, current.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		for _, execution := range executions {
			if execution.RoundID == nil || *execution.RoundID == *producer.RoundID || execution.RunID == nil || completed[execution.ExecutionID] || execution.State != auditstore.ExecutionSubmitted {
				continue
			}
			local, _ := h.artifacts.Run(*execution.RunID)
			taskRead, err := local.Read(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "task"})
			if err != nil {
				t.Fatal(err)
			}
			pkg, err := auditdomain.ValidatePackage(taskRead.Payload.Data)
			if err != nil {
				t.Fatal(err)
			}
			member, ok := pkg.MemberByID("task-document")
			if !ok {
				t.Fatal("typed task missing")
			}
			task, err := auditdomain.DecodeItemTask(member.Data())
			if err != nil || task.AcceptedPlan == nil || task.Dependencies == nil {
				t.Fatalf("typed task plan/closure: %+v %v", task, err)
			}
			if task.AcceptedPlan.Source.ReceiptID != receipt.ReceiptID || task.AcceptedPlan.Source.RunID != *producer.RunID || task.AcceptedPlan.Route.WorkflowRole != execution.WorkflowRole {
				t.Fatal("typed source/route pins drifted")
			}
			methods[task.AcceptedPlan.Plan.Method] = execution.WorkflowRole
			assertTypedConsumerMappings(t, ctx, h, *execution.RunID)
			finishTypedRoutingCheck(t, ctx, h, execution)
			completed[execution.ExecutionID] = true
		}
		current, err = h.audits.Get(ctx, current.OwnerID, current.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if len(completed) == 3 && current.OutstandingRunCount == 0 {
			break
		}
	}
	if !lost.lost || len(completed) != 3 || methods["verify"] != "blue" || methods["replay"] != "red" || methods["applicability"] != "green" {
		t.Fatalf("actual generic routes: lost=%t completed=%d methods=%+v state=%s stop=%+v", lost.lost, len(completed), methods, current.State, current.StopReason)
	}
	if current.SubmittedRunCount != 4 || current.ReservedRunCount != 4 || current.OutstandingRunCount != 0 {
		t.Fatalf("recovery changed target budget: submitted=%d reserved=%d outstanding=%d", current.SubmittedRunCount, current.ReservedRunCount, current.OutstandingRunCount)
	}
	progress := readTypedAdmissionProgress(t, ctx, h)
	if !progress.DeadlineAt.Equal(before.DeadlineAt) || progress.SelectedCount != 3 || progress.StagedCount != 3 || progress.Root == nil {
		t.Fatalf("restart changed accepted generation: %+v", progress)
	}
	for range 8 {
		if _, err := c.RunOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	var accepted, rejected, unknownItems int
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FILTER(WHERE source_provenance->>'disposition'='admitted'),count(*) FILTER(WHERE source_provenance->>'disposition'='rejected') FROM audit_artifact_links WHERE audit_id=$1 AND source_provenance->>'schema'=$2 AND source_provenance->>'receipt_id'=$3`, current.AuditID, auditstore.AdmissionDecisionSchema, receipt.ReceiptID).Scan(&accepted, &rejected); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(ctx, `SELECT count(*) FROM audit_items WHERE audit_id=$1 AND proposal_receipt_id=$2 AND proposal_check_ordinal=3`, current.AuditID, receipt.ReceiptID).Scan(&unknownItems); err != nil {
		t.Fatal(err)
	}
	if accepted != 3 || rejected != 1 || unknownItems != 0 {
		t.Fatalf("consume-once decisions: admitted=%d rejected=%d unknown items=%d", accepted, rejected, unknownItems)
	}
}

func readTypedAdmissionProgress(t *testing.T, ctx context.Context, h *postgresControllerHarness) auditstore.AdmissionProgress {
	t.Helper()
	var data []byte
	if err := h.pool.QueryRow(ctx, `SELECT admission_checkpoint FROM audit_rounds WHERE audit_id=$1 AND ordinal=1`, h.started.Audit.AuditID).Scan(&data); err != nil {
		t.Fatal(err)
	}
	if data == nil {
		return auditstore.AdmissionProgress{}
	}
	var result auditstore.AdmissionProgress
	if err := strictjson.Decode(data, &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func finishTypedRoutingCheck(t *testing.T, ctx context.Context, h *postgresControllerHarness, execution auditstore.Execution) {
	t.Helper()
	members, err := h.audits.ListExecutionItems(ctx, execution.ExecutionID)
	if err != nil || len(members) != 1 {
		t.Fatalf("check member: %+v %v", members, err)
	}
	items, err := h.audits.ListItemsByIDs(ctx, execution.AuditID, []string{members[0].ItemID})
	if err != nil || len(items) != 1 {
		t.Fatal(err)
	}
	project, _ := h.artifacts.Project(h.started.Audit.ProjectID)
	read, err := project.ReadExact(ctx, members[0].Task)
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := auditdomain.ValidatePackage(read.Payload.Data)
	if err != nil {
		t.Fatal(err)
	}
	member, _ := pkg.MemberByID("task-document")
	task, err := auditdomain.DecodeItemTask(member.Data())
	if err != nil {
		t.Fatal(err)
	}
	requested, assessment := []string{}, "satisfied"
	if task.Finding != nil {
		requested = []string{task.Finding.Method}
		assessment = "supported"
	}
	data, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: execution.Manifest.Digest, Results: []auditdomain.CheckResult{{ItemKey: task.ItemKey, SubjectKey: task.SubjectKey, Assessment: assessment, Summary: "The exact fixture check completed.", EvidenceIDs: []string{}, Coverage: auditdomain.ResultCoverage{Requested: requested, Completed: append([]string{}, requested...), Gaps: []string{}}, Proposals: []auditdomain.ProposalSelection{}}}})
	if err != nil {
		t.Fatal(err)
	}
	payload, _, err := auditdomain.BuildPackage("typed-fixture-result", auditdomain.PackageKindCheckResults, "", []auditdomain.PackageInput{{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: data}})
	if err != nil {
		t.Fatal(err)
	}
	local, _ := h.artifacts.Run(*execution.RunID)
	written, err := local.Write(ctx, contracts.ArtifactRef{Namespace: "worker", Name: "result"}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifactstest.BindOutput(ctx, h.artifacts, *execution.RunID, "result", written.Ref); err != nil {
		t.Fatal(err)
	}
	if err := artifactstest.FreezeOutputs(ctx, h.artifacts, *execution.RunID); err != nil {
		t.Fatal(err)
	}
	run, err := h.runs.GetRun(ctx, *execution.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if run.State == runstore.RunPending {
		if _, err := h.runs.TransitionRun(ctx, *execution.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture-admitted"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := h.runs.TransitionRun(ctx, *execution.RunID, runstore.RunRunning, runstore.RunSucceeded, runstore.Reason{Code: "fixture-complete"}); err != nil {
		t.Fatal(err)
	}
}

func assertTypedConsumerMappings(t *testing.T, ctx context.Context, h *postgresControllerHarness, runID string) {
	t.Helper()
	local, _ := h.artifacts.Run(runID)
	project, _ := h.artifacts.Project(h.started.Audit.ProjectID)
	read, err := local.Read(ctx, contracts.ArtifactRef{Namespace: "audit-context", Name: "retained-dependencies"})
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Mappings []struct {
			Name                       string
			Source, Retained, Consumer auditstore.ExactArtifact
		}
	}
	if err := json.Unmarshal(read.Payload.Data, &document); err != nil || len(document.Mappings) != 2 {
		t.Fatalf("consumer mapping: %s %v", read.Payload.Data, err)
	}
	for _, mapping := range document.Mappings {
		source, err := project.ReadExact(ctx, mapping.Retained)
		if err != nil {
			t.Fatal(err)
		}
		consumer, err := local.ReadExact(ctx, mapping.Consumer)
		if err != nil || string(source.Payload.Data) != string(consumer.Payload.Data) || mapping.Source.Digest != mapping.Retained.Digest || mapping.Retained.Digest != mapping.Consumer.Digest {
			t.Fatalf("exact retained consumer %s: %v", mapping.Name, err)
		}
	}
}
