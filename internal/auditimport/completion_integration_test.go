//go:build integration

package auditimport

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/httpapi/privateartifacts"
	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/mtls"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// The bridge runs the pinned ADK Runner in a separate Python process. Inputs,
// output bytes, grants and collection receipts all use production Go services.
// Scheduler driving and the external model are deterministic test orchestration.
func TestAuditCompletionRuntimeZIPImporter(t *testing.T) {
	pool := completionPool(t)
	for _, mode := range []string{"reverse", "correction", "dropped-reply", "missing", "partial", "invalid", "injected-empty-evidence", "injected-partial", "cancel-before-write", "cancel-after-write"} {
		t.Run(mode, func(t *testing.T) {
			f := newCompletionFixture(t, pool)
			bridge := f.bridge(t)
			probeMode := mode
			if strings.HasPrefix(mode, "injected-") || mode == "dropped-reply" {
				probeMode = "reverse"
			}
			if mode == "dropped-reply" {
				bridge.dropReply.Store(true)
			}
			report := bridge.probe(t, probeMode, false)
			if mode == "dropped-reply" && bridge.dropReply.Load() {
				t.Fatal("write-reply fault was not exercised")
			}
			value, err := f.output()
			switch mode {
			case "missing", "partial", "invalid", "cancel-before-write":
				if !errors.Is(err, artifacts.ErrArtifactNotFound) {
					t.Fatal("incomplete/cancelled invocation published a package", err)
				}
				outcome := runstore.RunFailed
				if mode == "cancel-before-write" {
					outcome = runstore.RunCancelled
				}
				f.collect(t, outcome, false)
			case "cancel-after-write":
				mustCompletion(t, err)
				if report["outcome"] != "cancelled" {
					t.Fatal("cancellation unexpectedly completed")
				}
				f.collect(t, runstore.RunCancelled, false)
			default:
				mustCompletion(t, err)
				if report["digest"] != completionDigest(value.Payload.Data) {
					t.Fatal("importer source differs from Runtime-published bytes")
				}
				if strings.HasPrefix(mode, "injected-") {
					f.injectInvalid(t, value, mode)
				}
				f.collect(t, runstore.RunSucceeded, !strings.HasPrefix(mode, "injected-"))
			}
		})
	}
	t.Run("process-loss-same-run-and-new-child", func(t *testing.T) {
		f := newCompletionFixture(t, pool)
		bridge := f.bridge(t)
		bridge.probe(t, "crash", false)
		before, err := f.output()
		mustCompletion(t, err)
		f.bridge(t).probe(t, "reverse", false)
		f.bridge(t).probe(t, "changed", true)
		f.bridge(t).probe(t, "proposal", true)
		after, err := f.output()
		mustCompletion(t, err)
		if !bytes.Equal(before.Payload.Data, after.Payload.Data) || *before.Ref.Revision != *after.Ref.Revision {
			t.Fatal("same-Run retry overwrote the create-only output")
		}
		// Fail this child and submit attempt 2 in the SAME Audit/round, through
		// CreateAudit again. Identical logical output names are Run-scoped.
		f.collect(t, runstore.RunFailed, false)
		f.createRun(t, 2)
		if _, err := f.output(); !errors.Is(err, artifacts.ErrArtifactNotFound) {
			t.Fatal("fresh child Run inherited a prior output", err)
		}
		retry := f.bridge(t)
		retry.probe(t, "changed", false)
		f.collect(t, runstore.RunSucceeded, true)
	})
	t.Run("process-loss-proposal-invocation-conflict", func(t *testing.T) {
		f := newCompletionFixture(t, pool)
		bridge := f.bridge(t)
		bridge.probe(t, "proposal-crash", false)
		before, err := f.output()
		mustCompletion(t, err)
		// Same client keys in a fresh invocation produce different proposal IDs.
		f.bridge(t).probe(t, "proposal", true)
		after, err := f.output()
		mustCompletion(t, err)
		if !bytes.Equal(before.Payload.Data, after.Payload.Data) || *before.Ref.Revision != *after.Ref.Revision {
			t.Fatal("proposal retry overwrote the original exact package")
		}
		f.collect(t, runstore.RunFailed, false)
	})
}

type completionFixture struct {
	ctx                       context.Context
	pool                      *pgxpool.Pool
	id                        string
	artifacts                 *artifacts.Service
	audits                    *auditstore.PostgresStore
	runs                      *runstore.PostgresStore
	service                   *runservice.Service
	profile                   config.ResolvedAuditProfile
	claim                     auditstore.ControllerClaim
	manifest, taskSet, source auditstore.ExactArtifact
	tasks                     []auditstore.ExactArtifact
	keys                      []string
	run                       runstore.WorkflowRun
	execution                 auditstore.Execution
}

func newCompletionFixture(t *testing.T, pool *pgxpool.Pool) *completionFixture {
	return newCompletionFixtureWithReportAcceptance(t, pool, "automatic")
}

func newCompletionFixtureWithReportAcceptance(t *testing.T, pool *pgxpool.Pool, acceptance string) *completionFixture {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)
	f := &completionFixture{ctx: ctx, pool: pool, id: fmt.Sprintf("fixture-%d", time.Now().UnixNano()), artifacts: artifacts.NewService(artifacts.NewPostgresRepository(pool)), audits: auditstore.NewPostgresStore(pool), runs: runstore.NewPostgresStore(pool), keys: []string{"first", "second"}}
	projects := projectstore.NewPostgresStore(pool)
	_, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: f.id, OwnerID: "owner", Kind: projectstore.KindProject, Name: "Completion gate", IdempotencyKey: f.id, RequestDigest: completionDigest([]byte(f.id))})
	mustCompletion(t, err)
	store, err := f.artifacts.Project(f.id)
	mustCompletion(t, err)
	write := func(name, media string, data []byte) auditstore.ExactArtifact {
		value, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "fixture", Name: name}, artifacts.Payload{MediaType: media, Data: data}, nil)
		mustCompletion(t, err)
		return auditstore.ExactArtifact{Ref: value.Ref, Digest: completionDigest(data), MediaType: media, SizeBytes: int64(len(data))}
	}
	f.source = write("source", "application/zip", []byte("scripted source"))
	checklist := []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"first","version":"1","statement":"First check","applicability":"always","allowed_methods":["static"],"required_evidence":["source-trace"],"review_policy":"automatic"},{"key":"second","version":"1","statement":"Second check","applicability":"always","allowed_methods":["static"],"required_evidence":["source-trace"],"review_policy":"automatic"}]}`)
	checklistRef := write("checklist", "application/json", checklist)
	inventory, err := auditdomain.BuildChecklistInventory(checklist, "application/json", auditdomain.InventoryOptions{Round: 1, WorkflowRole: "check", SourceInputName: "source", SourceRef: f.source.Ref, ApprovalRequirement: auditdomain.ApprovalNone})
	mustCompletion(t, err)
	var members []auditdomain.PackageInput
	var items []auditstore.MaterializedItem
	for i, task := range inventory.Tasks {
		descriptor := write(f.keys[i], "application/zip", task.Package)
		f.tasks = append(f.tasks, descriptor)
		inventory.ExecutionManifest.Items[i].TaskRef = &descriptor.Ref
		members = append(members, auditdomain.PackageInput{ID: fmt.Sprintf("task-%03d", i), Path: fmt.Sprintf("tasks/%03d.zip", i), MediaType: "application/zip", Data: task.Package})
		items = append(items, auditstore.MaterializedItem{ItemID: f.id + "-" + f.keys[i], ItemKey: f.keys[i], Ordinal: i, Kind: task.Document.Kind, SubjectKey: inventory.ExecutionManifest.Items[i].SubjectKey, Task: descriptor,
			Origin:       auditstore.ItemOrigin{Schema: auditstore.ItemOriginSchema, SourceRef: &checklistRef.Ref, SourceContentDigest: inventory.SourceContentDigest, SourceMediaType: "application/json", CanonicalInventoryDigest: inventory.CanonicalInventoryDigest, EntryKey: f.keys[i], EntryVersion: "1"},
			WorkflowRole: "check", InitialState: auditstore.ItemReady, Coverage: auditstore.Coverage{Status: auditstore.CoverageNotTested, Requested: []string{"source-trace"}, Completed: []string{}, Gaps: []string{}}})
	}
	manifest, err := auditdomain.EncodeExecutionManifest(inventory.ExecutionManifest)
	mustCompletion(t, err)
	f.manifest = write("manifest", "application/json", manifest)
	batch, _, err := auditdomain.BuildPackage("task-set", auditdomain.PackageKindTaskSet, "", members)
	mustCompletion(t, err)
	f.taskSet = write("tasks", "application/zip", batch)
	// Load a temporary authored closure so its digest is verified by the importer.
	// Remove only the fixture's unused source-analysis Skill dependency.
	catalogRoot := t.TempDir()
	mustCompletion(t, os.CopyFS(catalogRoot, os.DirFS("../../configs")))
	for _, name := range []string{
		"agent-templates/audit_source_checker.yaml",
		"workflows/audit_source_check.yaml",
		"audit-profiles/source_checklist.yaml",
		"model-policies/audit_completion_worker.yaml",
		"instructions/audit-source-checker-worker.md",
	} {
		data, err := os.ReadFile(filepath.Join("../../configs", name))
		mustCompletion(t, err)
		if strings.HasPrefix(name, "agent-templates/") {
			data = bytes.ReplaceAll(data, []byte("  skills:\n    - namespace: skills\n      name: trace\n"), nil)
		}
		if strings.HasPrefix(name, "audit-profiles/") {
			data = bytes.ReplaceAll(data, []byte("reportAcceptance: automatic"), []byte("reportAcceptance: "+acceptance))
		}
		mustCompletion(t, os.WriteFile(filepath.Join(catalogRoot, name), data, 0600))
	}
	catalog, err := config.Load(catalogRoot, config.MVPDescriptors())
	mustCompletion(t, err)
	f.profile, err = catalog.AuditProfile("source-checklist@1")
	mustCompletion(t, err)
	profileBytes, err := json.Marshal(f.profile)
	mustCompletion(t, err)
	_, err = config.DecodeResolvedAuditProfileSnapshot(profileBytes)
	mustCompletion(t, err)
	draft, _, err := f.audits.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: f.id, OwnerID: "owner", ProjectID: f.id, Profile: auditstore.ProfileIdentity{Name: f.profile.Ref.Name, Version: f.profile.Ref.Version, Digest: f.profile.Ref.Digest}, ProfileSnapshot: profileBytes, InputSelection: json.RawMessage(`{}`), Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 2, MaxItemsPerRound: 10, MaxItemsTotal: 10, MaxSubmittedRuns: 10, MaxItemRunAttempts: 2, MaxEvidenceBytes: 1 << 20}, IdempotencyKey: f.id, RequestDigest: completionDigest([]byte("audit"))})
	mustCompletion(t, err)
	_, _, err = f.audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{OwnerID: "owner", AuditID: f.id, ExpectedRevision: draft.Revision, RoundID: f.id, RoundOrdinal: 1, Manifest: f.manifest, BaselineSnapshot: json.RawMessage(`{"schema":"contractor.audit.baseline.v1","inventory":{"gaps":[]}}`), DeadlineAt: time.Now().Add(time.Hour), Items: items, IdempotencyKey: f.id, RequestDigest: completionDigest([]byte("round"))})
	mustCompletion(t, err)
	claims, err := f.audits.Claim(ctx, auditstore.ClaimParams{HolderID: f.id, Lease: 5 * time.Minute, Limit: 100})
	mustCompletion(t, err)
	for _, claim := range claims {
		if claim.AuditID == f.id {
			f.claim = claim
		}
	}
	if f.claim.AuditID == "" {
		t.Fatal("fixture Audit was not claimed")
	}
	_, err = f.audits.TransitionRound(ctx, auditstore.RoundTransitionParams{Claim: f.claim, RoundID: f.id, ExpectedRevision: 1, ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting})
	mustCompletion(t, err)
	manager, err := config.NewManager(config.ManagerOptions{OperatorRoot: "../config/testdata/valid", ManagedRoot: filepath.Join(t.TempDir(), "managed"), Descriptors: config.MVPDescriptors()})
	mustCompletion(t, err)
	f.service, err = runservice.New(runservice.Options{Runs: f.runs, Workflows: manager, LLMCredentials: completionCredentials{}, CredentialGuard: completionCredentials{}, RuntimeCredentials: completionCredentials{}, Projects: projects,
		PublicTransaction: func(context.Context, func(runservice.PublicRunWriter, *artifacts.Service) error) error {
			return errors.New("unexpected public creation")
		},
		AuditTransaction: func(ctx context.Context, fn func(runservice.AuditRunWriter, *artifacts.Service, runservice.AuditExecutionWriter) error) error {
			return persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				return fn(runstore.NewPostgresStore(tx), artifacts.NewService(artifacts.NewPostgresRepository(tx)), auditstore.NewPostgresStore(tx))
			})
		},
	})
	mustCompletion(t, err)
	f.createRun(t, 1)
	return f
}

func (f *completionFixture) createRun(t *testing.T, attempt int) {
	t.Helper()
	id := fmt.Sprintf("%s-attempt-%d", f.id, attempt)
	var members []auditstore.ExecutionMemberIntent
	for i, task := range f.tasks {
		members = append(members, auditstore.ExecutionMemberIntent{ExecutionItemID: id + "-" + f.keys[i], ItemID: f.id + "-" + f.keys[i], BatchOrdinal: i, ItemAttempt: attempt, Task: task, Inputs: []auditstore.ExactArtifact{f.source, f.taskSet}})
	}
	execution, _, err := f.audits.CreateExecutionIntent(f.ctx, auditstore.CreateExecutionIntentParams{Claim: f.claim, ExecutionID: id, RoundID: &f.id, Role: auditstore.ExecutionCheck, WorkflowRole: "check", Manifest: f.manifest, SubmissionKey: id, RequestDigest: completionDigest([]byte(id)), Members: members})
	mustCompletion(t, err)
	created, err := f.service.CreateAudit(f.ctx, runservice.AuditCreateParams{Claim: f.claim, ExecutionID: id, Workflow: f.profile.Workflows["check"].Workflow, RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(), Parameters: map[string]string{}, Inputs: map[string]auditstore.ExactArtifact{"source": f.source, "task": f.taskSet, "execution_manifest": f.manifest}, ExecutionManifest: f.manifest, RequestDigest: execution.RequestDigest, NewRunID: func() (string, error) { return id, nil }})
	mustCompletion(t, err)
	// This fixture drives execution directly, including Scheduler admission.
	created.Run, err = f.runs.TransitionRun(f.ctx, created.Run.RunID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "fixture-admitted"})
	mustCompletion(t, err)
	f.run, f.execution = created.Run, execution
	if f.run.AuditCompletion == nil {
		t.Fatal("trusted Run creation omitted completion contract")
	}
	// Moving the mutable alias must not change the exact input used by Runtime.
	store, err := f.artifacts.Run(f.run.RunID)
	mustCompletion(t, err)
	_, err = store.Write(f.ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "task"}, artifacts.Payload{MediaType: "application/zip", Data: []byte("changed alias")}, f.run.AuditCompletion.Contract.Task.Revision)
	mustCompletion(t, err)
}

func (f *completionFixture) output() (artifacts.ReadResult, error) {
	store, err := f.artifacts.Run(f.run.RunID)
	if err != nil {
		return artifacts.ReadResult{}, err
	}
	return store.Read(f.ctx, f.run.AuditCompletion.Contract.ResultArtifact)
}

func (f *completionFixture) collect(t *testing.T, outcome runstore.WorkflowRunState, accepted bool) {
	t.Helper()
	if outcome == runstore.RunSucceeded {
		value, err := f.output()
		mustCompletion(t, err)
		_, err = f.artifacts.BindOutputExact(f.ctx, f.run.RunID, f.profile.Workflows["check"].Outputs["result"], value.Ref, nil)
		mustCompletion(t, err)
	}
	mustCompletion(t, f.artifacts.FreezeRunOutputs(f.ctx, f.run.RunID))
	expected := runstore.RunRunning
	if outcome == runstore.RunCancelled {
		_, err := f.runs.RequestRunCancellation(f.ctx, f.run.RunID, runstore.WorkflowRunCancellation{Code: runstore.CancellationUserRequested, RequestedAt: time.Now().UTC()})
		mustCompletion(t, err)
		expected = runstore.RunCancelling
	}
	_, err := f.runs.TransitionRun(f.ctx, f.run.RunID, expected, outcome, runstore.Reason{Code: "fixture-terminal"})
	mustCompletion(t, err)
	cursor, err := f.runs.GetRunEventCursor(f.ctx, f.run.RunID)
	mustCompletion(t, err)
	execution, err := f.audits.ObserveTerminal(f.ctx, auditstore.ObserveTerminalParams{Claim: f.claim, ExecutionID: f.execution.ExecutionID, RunID: f.run.RunID, Generation: cursor.Generation, Sequence: uint64(cursor.Sequence)})
	mustCompletion(t, err)
	snapshot, err := f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	access, err := NewArtifactAccess(f.artifacts)
	mustCompletion(t, err)
	importer, err := New(f.audits, f.runs, access)
	mustCompletion(t, err)
	changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
	mustCompletion(t, err)
	if !changed {
		t.Fatal("importer did not commit collection")
	}
	rows, err := f.audits.ListCoverage(f.ctx, f.id, f.id, -1, 10)
	mustCompletion(t, err)
	if len(rows) != 2 {
		t.Fatalf("coverage rows = %d", len(rows))
	}
	for _, row := range rows {
		if accepted != (row.Result != nil && row.Coverage.Status == auditstore.CoverageSatisfied) {
			t.Fatalf("unexpected coverage: %+v", row)
		}
		if !accepted && row.Result != nil {
			t.Fatal("failed batch partially accepted")
		}
	}
	snapshot, err = f.audits.GetReconcileSnapshot(f.ctx, f.claim)
	mustCompletion(t, err)
	var receipt *auditstore.CollectionReceiptSummary
	for i := range snapshot.Receipts {
		if snapshot.Receipts[i].ExecutionID == execution.ExecutionID {
			receipt = &snapshot.Receipts[i]
		}
	}
	if receipt == nil || accepted != (receipt.Disposition == auditstore.CollectionAccepted) {
		t.Fatalf("unexpected receipt: %+v", receipt)
	}
	if outcome == runstore.RunSucceeded && !accepted && receipt.Disposition != auditstore.CollectionInvalidResult {
		t.Fatalf("injected packet not independently rejected: %+v", receipt)
	}
}

func (f *completionFixture) injectInvalid(t *testing.T, value artifacts.ReadResult, mode string) {
	t.Helper()
	original, err := auditdomain.DecodeCheckResultPackage(value.Payload.Data)
	mustCompletion(t, err)
	pkg := original.Package
	removedEvidence := make(map[string]bool)
	removedContent := make(map[string]bool)
	for _, id := range original.Results.Results[1].EvidenceIDs {
		removedEvidence[id] = true
	}
	for _, evidence := range original.Evidence.Evidence {
		if removedEvidence[evidence.ID] {
			removedContent[evidence.ContentMemberID] = true
		}
	}
	var inputs []auditdomain.PackageInput
	for _, member := range pkg.Manifest.Members {
		if removedContent[member.ID] {
			continue
		}
		value, ok := pkg.Member(member.Path)
		if !ok {
			t.Fatal("missing package member")
		}
		data := value.Data()
		if member.Path == "check-results.json" {
			var document map[string]any
			mustCompletion(t, json.Unmarshal(data, &document))
			results := document["results"].([]any)
			if mode == "injected-partial" {
				document["results"] = results[:1]
			} else {
				results[1].(map[string]any)["evidence_ids"] = []string{}
			}
			data, err = json.Marshal(document)
			mustCompletion(t, err)
		}
		if member.Path == "evidence.json" {
			var document map[string]any
			mustCompletion(t, json.Unmarshal(data, &document))
			kept := []any{}
			for _, evidence := range document["evidence"].([]any) {
				if !removedEvidence[evidence.(map[string]any)["id"].(string)] {
					kept = append(kept, evidence)
				}
			}
			document["evidence"] = kept
			data, err = json.Marshal(document)
			mustCompletion(t, err)
		}
		inputs = append(inputs, auditdomain.PackageInput{ID: member.ID, Path: member.Path, MediaType: member.MediaType, Data: data})
	}
	data, _, err := auditdomain.BuildPackage(pkg.Manifest.PackageID, pkg.Manifest.Kind, pkg.Manifest.EntryPoint, inputs)
	mustCompletion(t, err)
	// The injected ZIP remains structurally valid: the importer must reject
	// task evidence/membership semantics, not orphaned members or corrupt hashes.
	_, err = auditdomain.DecodeCheckResultPackage(data)
	mustCompletion(t, err)
	store, err := f.artifacts.Run(f.run.RunID)
	mustCompletion(t, err)
	_, err = store.Write(f.ctx, f.run.AuditCompletion.Contract.ResultArtifact, artifacts.Payload{MediaType: "application/zip", Data: data}, value.Ref.Revision)
	mustCompletion(t, err)
}

type completionBridge struct {
	f         *completionFixture
	spec      map[string]any
	dropReply atomic.Bool
	release   func()
}

func (f *completionFixture) bridge(t *testing.T) *completionBridge {
	t.Helper()
	root := filepath.Join(t.TempDir(), "pki")
	g := localpki.Generator{}
	ca, err := g.InitCA(root, false)
	mustCompletion(t, err)
	leaf := localpki.LeafOptions{DNSNames: []string{"localhost"}, IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}}
	serverCert, err := g.IssueControlPlane(root, localpki.ControlPlaneOptions{LeafOptions: leaf})
	mustCompletion(t, err)
	agentCert, err := g.IssueAgent(root, "completion-agent", leaf)
	mustCompletion(t, err)
	data, err := os.ReadFile(agentCert.Certificate)
	mustCompletion(t, err)
	block, _ := pem.Decode(data)
	if block == nil {
		t.Fatal("invalid certificate")
	}
	cert, err := x509.ParseCertificate(block.Bytes)
	mustCompletion(t, err)
	principal, err := mtls.RuntimeAgentID(cert)
	mustCompletion(t, err)
	registry, err := controlplane.NewRegistry(controlplane.RegistryOptions{HeartbeatInterval: time.Minute, ConfirmedLease: 5 * time.Minute})
	mustCompletion(t, err)
	stage := f.profile.Workflows["check"].Workflow.Stages["check"]
	template := stage.Agents["checker"].Template
	toolsets := make([]contracts.ToolsetCapability, 0, len(template.Toolsets))
	for _, set := range template.Toolsets {
		toolsets = append(toolsets, contracts.ToolsetCapability{Ref: set.Ref.ToolsetID + "@" + set.Ref.Version, Tools: set.Tools})
	}
	_, err = registry.RegisterAuthenticated(controlplane.AuthenticatedPrincipal{RuntimeAgentID: principal, Labels: []string{}, LabelRevision: 1}, contracts.AgentRegistration{APIVersion: contracts.APIVersion, InstanceID: "runtime-1", SoftwareVersion: "fixture", StartedAt: time.Now().UTC(), ControlURL: "https://localhost:9443", A2AURL: "https://localhost:9444", InitialLabels: []string{}, SupportedRuntimeAdapters: []contracts.RuntimeAdapterRef{}, SupportedRuntimes: []string{"adk@1"}, SupportedToolsets: toolsets, SupportedSandboxProfiles: []string{"local-workdir@1"}, ObservedState: contracts.AgentIdle, Capabilities: &contracts.RuntimeCompletionCapabilities{CompletionContracts: []string{contracts.AuditCheckResultsV1}}})
	mustCompletion(t, err)
	for seq := uint64(1); seq <= 2; seq++ {
		_, err = registry.HeartbeatAuthenticated(principal, contracts.AgentHeartbeat{APIVersion: contracts.APIVersion, InstanceID: "runtime-1", HeartbeatSeq: seq, EchoedAckSeq: seq - 1, ObservedState: contracts.AgentIdle})
		mustCompletion(t, err)
	}
	reservation, err := registry.ReserveAll(controlplane.ReservationRequest{RunID: f.run.RunID, StageExecutionID: f.run.RunID + "-stage", Bindings: []controlplane.BindingRequirement{{LogicalAgentName: "checker", Namespace: f.run.AuditCompletion.Contract.ResultArtifact.Namespace, WorkerSessionMode: contracts.WorkerSessionShared, AgentTemplate: template, CompletionContract: &f.run.AuditCompletion.Contract, ExecutionConfig: controlplane.AllocationExecutionConfig{ModelPolicy: template.ModelPolicy.Ref, LLMGateway: stage.ExecutionConfig.Agents["checker"].LLMGateway.Ref}}}})
	mustCompletion(t, err)
	if len(reservation) != 1 {
		t.Fatal("missing reservation")
	}
	id := reservation[0].Grant.AllocationID
	var released sync.Once
	release := func() {
		released.Do(func() {
			mustCompletion(t, registry.SetWriteFence(id))
			mustCompletion(t, registry.Release(id))
			agent, err := registry.GetAgent("runtime-1")
			mustCompletion(t, err)
			if agent.AuthoritativeAllocationID != nil {
				t.Error("allocation slot leaked")
			}
		})
	}
	t.Cleanup(release)
	handler, err := privateartifacts.NewHandler(privateartifacts.Dependencies{Registry: registry, Artifacts: f.artifacts})
	mustCompletion(t, err)
	b := &completionBridge{f: f, release: release}
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && b.dropReply.Load() {
			recorded := httptest.NewRecorder()
			handler.ServeHTTP(recorded, r)
			if recorded.Code >= 200 && recorded.Code < 300 && b.dropReply.CompareAndSwap(true, false) {
				connection, _, err := w.(http.Hijacker).Hijack()
				if err == nil {
					_ = connection.Close()
				}
				return
			}
			for key, values := range recorded.Header() {
				w.Header()[key] = values
			}
			w.WriteHeader(recorded.Code)
			_, _ = w.Write(recorded.Body.Bytes())
			return
		}
		handler.ServeHTTP(w, r)
	}))
	server.TLS, err = mtls.ControlPlaneServerConfig(mtls.Files{Certificate: serverCert.Certificate, PrivateKey: serverCert.PrivateKey, CA: ca.Certificate})
	mustCompletion(t, err)
	server.StartTLS()
	t.Cleanup(server.Close)
	b.spec = map[string]any{"apiURL": server.URL + "/private/v1", "ca": ca.Certificate, "certificate": agentCert.Certificate, "privateKey": agentCert.PrivateKey, "allocationID": id, "runID": f.run.RunID, "contract": f.run.AuditCompletion.Contract, "itemKeys": f.keys}
	return b
}

func (b *completionBridge) probe(t *testing.T, mode string, conflict bool) map[string]any {
	t.Helper()
	defer b.release() // Server reconciliation releases even a crashed process's slot.
	dir := t.TempDir()
	report := filepath.Join(dir, "result.json")
	b.spec["mode"], b.spec["expectConflict"], b.spec["report"] = mode, conflict, report
	data, err := json.Marshal(b.spec)
	mustCompletion(t, err)
	spec := filepath.Join(dir, "spec.json")
	mustCompletion(t, os.WriteFile(spec, data, 0600))
	root, err := filepath.Abs("../..")
	mustCompletion(t, err)
	ctx, cancel := context.WithTimeout(b.f.ctx, 45*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, filepath.Join(root, "runtime/.venv/bin/python"), "-m", "pytest", "tests/test_audit_completion_e2e.py", "-q", "--junitxml", filepath.Join(dir, "runtime.xml"))
	command.Dir = filepath.Join(root, "runtime")
	command.Env = append(os.Environ(), "CONTRACTOR_AUDIT_COMPLETION_BRIDGE="+spec)
	output, err := command.CombinedOutput()
	if mode == "crash" || mode == "proposal-crash" {
		var failure *exec.ExitError
		if !errors.As(err, &failure) || failure.ExitCode() != 73 {
			t.Fatalf("expected abrupt post-write process loss: %v\n%s", err, output)
		}
		if _, err := os.Stat(report); !os.IsNotExist(err) {
			t.Fatal("crashed Runtime reported completion")
		}
		t.Logf("Runtime process %s: lost after write, exit=73, no completion receipt", mode)
		return nil
	}
	if err != nil {
		t.Fatalf("Runtime probe %s: %v\n%s", mode, err, output)
	}
	data, err = os.ReadFile(report)
	mustCompletion(t, err) // A skipped/empty pytest run cannot pass.
	var value map[string]any
	mustCompletion(t, json.Unmarshal(data, &value))
	if value["outcome"] == nil {
		t.Fatal("Runtime probe omitted its outcome")
	}
	t.Logf("Runtime process %s: outcome=%s modelCalls=%v", mode, value["outcome"], value["modelCalls"])
	return value
}

func completionPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required for the Runtime/importer bridge")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	configuration, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		t.Fatal("invalid test PostgreSQL configuration")
	}
	admin, err := pgxpool.NewWithConfig(ctx, configuration)
	if err != nil {
		t.Fatal("cannot initialize test PostgreSQL")
	}
	t.Cleanup(admin.Close)
	if err := admin.Ping(ctx); err != nil {
		t.Fatal("test PostgreSQL is unavailable")
	}
	var suffix [8]byte
	_, err = rand.Read(suffix[:])
	mustCompletion(t, err)
	name := "contractor_completion_" + hex.EncodeToString(suffix[:])
	_, err = admin.Exec(ctx, `CREATE DATABASE `+pgx.Identifier{name}.Sanitize())
	mustCompletion(t, err)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, err := admin.Exec(ctx, `DROP DATABASE `+pgx.Identifier{name}.Sanitize()+` WITH (FORCE)`)
		if err != nil {
			t.Error("cannot remove disposable completion database")
		}
	})
	configuration = configuration.Copy()
	configuration.ConnConfig.Database = name
	pool, err := pgxpool.NewWithConfig(ctx, configuration)
	mustCompletion(t, err)
	t.Cleanup(pool.Close)
	_, err = persistencepostgres.ApplyMigrations(ctx, pool)
	mustCompletion(t, err)
	return pool
}

func completionDigest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}
func mustCompletion(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

type completionCredentials struct{}

func (completionCredentials) LookupLLMCredential(context.Context, string) (config.CredentialMetadata, error) {
	return config.CredentialMetadata{}, errors.New("no live credentials in completion fixtures")
}
func (completionCredentials) WithRunCreation(_ context.Context, fn func() error) error { return fn() }
func (completionCredentials) ValidateRuntimeCredential(context.Context, string, ...string) error {
	return nil
}
