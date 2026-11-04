package auditservice

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

func checkingProgram(p *auditcontract.Program) {
	p.Workflows["inspect"] = auditcontract.ProgramWorkflow{Ref: "audit-contract-check@1", Inputs: map[string]auditcontract.InputMapping{"source": {Source: "audit-input", Name: "source"}, "context": {Source: "task-context"}}, Parameters: map[string]string{}, Outputs: map[string]string{"result": "result"}}
	p.Hypotheses = map[string]auditcontract.HypothesisSeed{"fixed-one": {Statement: "Reader A can access B's record under the selected conditions", Scope: map[string]string{"actor": "reader A"}, Grounds: "Selected checklist assertion"}, "fixed-two": {Statement: "Only an owner can access their record under these conditions", Scope: map[string]string{"actor": "owner"}, Grounds: "Selected positive property"}}
	p.InitialWork = []auditcontract.InitialWork{{Key: "first", Workflow: "inspect", Objective: "Check selected assertion", Hypothesis: "fixed-one"}, {Key: "second", Workflow: "inspect", Objective: "Check selected property", Hypothesis: "fixed-two"}}
}

func TestSharedHypothesesInitialAndManualExactPins(t *testing.T) {
	f := newCommonTaskServiceFixture(t, checkingProgram)
	ctx := t.Context()
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='active',dispatch_state='open' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	claims, err := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage()).Claim(ctx, auditstore.ClaimParams{HolderID: "hypothesis-test", Limit: 1, Lease: time.Minute})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claims=%+v error=%v", claims, err)
	}
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET state='paused',dispatch_state='closed' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	snapshot := auditstore.ReconcileSnapshot{Audit: f.audit}
	intake, err := f.service.AcceptInitialTasks(ctx, claims[0], snapshot)
	if err != nil || !intake.Changed {
		t.Fatalf("seed intake=%+v %v", intake, err)
	}
	snapshot.Audit, err = f.service.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	intake, err = f.service.AcceptInitialTasks(ctx, claims[0], snapshot)
	if err != nil || !intake.Complete {
		t.Fatalf("intake=%+v error=%v", intake, err)
	}
	page, err := f.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 10})
	if err != nil || len(page.Items) != 2 {
		t.Fatalf("hypotheses=%+v error=%v", page, err)
	}
	tasks, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 10})
	if err != nil || len(tasks.Tasks) != 2 {
		t.Fatal(err)
	}
	first := page.Items[0]
	current, err := f.service.Get(ctx, "owner", f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	edit := HypothesisMutationParams{MutationParams: MutationParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: current.Revision, IdempotencyKey: "edit", RequestDigest: contentdigest.Bytes([]byte("edit"))}, HypothesisID: first.Document.ID, Content: HypothesisContent{Key: first.SourceKey, Statement: "Reader A can access another record in a different scope", Scope: first.Document.Scope, Grounds: "Explicit revised claim"}}
	changed, replayed, err := f.service.WriteHypothesis(ctx, edit)
	if err != nil || replayed || changed.Hypotheses[0].Revision != 2 {
		t.Fatalf("edit=%+v replay=%v error=%v", changed, replayed, err)
	}
	again, replayed, err := f.service.WriteHypothesis(ctx, edit)
	if err != nil || !replayed || again.AuditRevision != changed.AuditRevision {
		t.Fatalf("replay=%+v %v", again, err)
	}
	task, err := f.service.Task(ctx, "owner", f.audit.AuditID, tasks.Tasks[0].Document.TaskID)
	if err != nil || task.Document.Hypothesis.Revision != 1 || task.Hypothesis.Statement != first.Document.Statement {
		t.Fatalf("accepted pin changed: %+v %v", task, err)
	}
	old, err := f.service.Hypothesis(ctx, "owner", f.audit.AuditID, first.Document.ID, 1)
	if err != nil || old.Document.Statement != first.Document.Statement {
		t.Fatal("original assertion lost", err)
	}
	if _, err := f.service.Hypothesis(ctx, "foreign", f.audit.AuditID, first.Document.ID, 1); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign read=%v", err)
	}
	manual := HypothesisMutationParams{MutationParams: MutationParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: changed.AuditRevision, IdempotencyKey: "manual", RequestDigest: contentdigest.Bytes([]byte("manual"))}, Content: HypothesisContent{Key: "manual", Statement: "A manual observable claim", Scope: map[string]string{}, Grounds: "Analyst observation"}}
	accepted, _, err := f.service.WriteHypothesis(ctx, manual)
	if err != nil {
		t.Fatal(err)
	}
	work := []auditcontract.PreparedSelection{{Key: "manual-check", Workflow: "inspect", Objective: "Check manual assertion", Inputs: map[string]auditcontract.Material{}}}
	bound, err := f.service.CreateTasks(ctx, TaskCreationParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: accepted.AuditRevision, IdempotencyKey: "manual-task", RequestDigest: contentdigest.Bytes([]byte("manual-task")), Work: work, Hypotheses: map[string]string{"manual-check": accepted.Hypotheses[0].ID}})
	if err != nil {
		t.Fatal(err)
	}
	manualTask, err := f.service.Task(ctx, "owner", f.audit.AuditID, bound.Acceptance.TaskIDs[0])
	if err != nil || manualTask.Hypothesis.Origin.Kind != "manual" || manualTask.Document.Hypothesis.Revision != 1 {
		t.Fatalf("manual task=%+v error=%v", manualTask, err)
	}
}

func TestSharedAssertionSourcesFreezeRealArtifactsBeforeBoundedAdmission(t *testing.T) {
	for _, kind := range []string{"checklist", "standard"} {
		t.Run(kind, func(t *testing.T) {
			f := newCommonTaskServiceFixture(t, func(program *auditcontract.Program) {
				checkingProgram(program)
				program.Hypotheses = map[string]auditcontract.HypothesisSeed{}
				program.InitialWork = []auditcontract.InitialWork{}
				program.SeedSource = &auditcontract.SeedSource{Workflow: "inspect"}
				if kind == "checklist" {
					program.SeedSource.Input = "checklist"
					program.Inputs["checklist"] = auditcontract.ProgramInput{Required: true, MediaTypes: []string{"application/json"}}
				} else {
					program.SeedSource.Standard = &auditcontract.SeedStandard{Scheme: "example", Version: "1"}
				}
			})
			ctx := t.Context()
			art := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool))
			baseline, err := DecodeBaseline(f.audit.BaselineSnapshot)
			if err != nil {
				t.Fatal(err)
			}
			if kind == "checklist" {
				project, _ := art.Project(f.audit.ProjectID)
				data := []byte(`{"schema":"contractor.audit.assertions.v1","items":[{"key":"first","version":"1","statement":"The first scoped property holds.","scope":{},"grounds":"An authored source","applicability":"always","allowed_methods":["static"],"required_evidence":["Exact source"]},{"key":"second","version":"1","statement":"The second scoped property holds.","scope":{},"grounds":"An authored source","applicability":"always","allowed_methods":["static"],"required_evidence":["Exact source"]}]}`)
				written, err := project.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checklist"}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
				if err != nil {
					t.Fatal(err)
				}
				baseline.Inputs["checklist"] = contracts.ExactArtifact{Ref: written.Ref, Digest: contentdigest.Bytes(data), MediaType: "application/json", SizeBytes: int64(len(data))}
			} else {
				root := configtest.CopyWithPolicies(t, "../config/testdata/valid")
				if err := os.CopyFS(root, os.DirFS("testdata/hypothesis-standard")); err != nil {
					t.Fatal(err)
				}
				_, bundles, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
				if err != nil {
					t.Fatal(err)
				}
				catalog, err := auditstandards.NewCatalog(art)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := catalog.Initialize(ctx, "owner", bundles.Standards); err != nil {
					t.Fatal(err)
				}
				if _, err := catalog.Initialize(ctx, "foreign", bundles.Standards); err != nil {
					t.Fatal(err)
				}
				foreign, err := catalog.Resolve(ctx, "foreign", auditstandards.Reference{Scheme: "example", Version: "1"})
				if err != nil {
					t.Fatal(err)
				}
				source, _ := artifacts.UserScope("foreign")
				target, _ := artifacts.AuditScope(f.audit.AuditID)
				if _, err := art.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditInput, Source: source, SourceRef: foreign.Source.Artifact, Target: target, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "foreign-source"}, Frozen: true}); err == nil {
					t.Fatal("foreign User standard entered Audit scope")
				}

			}
			if err := pinAssertionSource(ctx, art, f.audit, f.pinned, &baseline); err != nil {
				t.Fatal(err)
			}
			encoded, err := EncodeBaseline(baseline)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.pool.Exec(ctx, `UPDATE audits SET baseline_snapshot=$2,max_items_total=1,max_items_per_round=1,batch_size=1,state='active',dispatch_state='open' WHERE audit_id=$1`, f.audit.AuditID, encoded); err != nil {
				t.Fatal(err)
			}
			store := auditstoretest.NewStore(f.pool, artifactpgtest.NewStorage())
			claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "source-controller", Lease: time.Minute, Limit: 1})
			if err != nil || len(claims) != 1 {
				t.Fatalf("claims=%+v error=%v", claims, err)
			}
			// Exercise the real activation owner, including refusal of a
			// different Audit's frozen source, before admitting any Tasks.
			draft, _, err := store.CreateDraft(ctx, auditstore.CreateDraftParams{AuditID: "activation-audit", OwnerID: "owner", ProjectID: f.audit.ProjectID, Profile: f.audit.Profile, ProfileSnapshot: f.audit.ProfileSnapshot, InputSelection: json.RawMessage(`{}`), Limits: f.audit.Limits, IdempotencyKey: "activation-draft", RequestDigest: contentdigest.Bytes([]byte("activation-draft"))})
			if err != nil {
				t.Fatal(err)
			}
			activation := auditstore.ActivateParams{OwnerID: "owner", AuditID: draft.AuditID, ExpectedRevision: draft.Revision, BaselineSnapshot: encoded, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "activation-start", RequestDigest: contentdigest.Bytes([]byte("activation-start"))}
			if _, _, err := ownerstore.New(f.pool, artifactpgtest.NewStorage()).Activate(ctx, activation); !errors.Is(err, auditstore.ErrInvalid) {
				t.Fatalf("foreign assertion source activation=%v", err)
			}
			ownBaseline := baseline
			ownBaseline.Standards = []auditstandards.PinnedPackage{}
			if err := pinAssertionSource(ctx, art, draft, f.pinned, &ownBaseline); err != nil {
				t.Fatal(err)
			}
			activation.BaselineSnapshot, err = EncodeBaseline(ownBaseline)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := ownerstore.New(f.pool, artifactpgtest.NewStorage()).Activate(ctx, activation); err != nil {
				t.Fatal(err)
			}
			// All selected hypotheses are admitted even though the Task budget is one.
			for range 3 {
				audit, err := f.service.Get(ctx, "owner", f.audit.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				_, err = f.service.AcceptInitialTasks(ctx, claims[0], auditstore.ReconcileSnapshot{Audit: audit})
				if err != nil {
					t.Fatal(err)
				}
			}
			hypotheses, err := f.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 10})
			if err != nil || len(hypotheses.Items) != 2 || hypotheses.FixedDenominator != 2 {
				t.Fatalf("coverage=%+v error=%v", hypotheses, err)
			}
			tasks, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 10})
			if err != nil || len(tasks.Tasks) != 1 {
				t.Fatalf("tasks=%+v error=%v", tasks, err)
			}
			if tasks.Tasks[0].Hypothesis == nil || tasks.Tasks[0].Hypothesis.Origin.Source == nil || !tasks.Tasks[0].Hypothesis.Origin.Source.Ref.SameExact(baseline.InitialAssertions.Source.Ref) {
				t.Fatal("lost exact retained source")
			}
			scope, _ := artifacts.AuditScope(f.audit.AuditID)
			retained, _ := art.Scoped(scope)
			read, err := retained.Read(ctx, baseline.InitialAssertions.Source.Ref)
			if err != nil || contentdigest.Bytes(read.Payload.Data) != baseline.InitialAssertions.Source.Digest {
				t.Fatal("unreadable source", err)
			}
			var runs int
			if err := f.pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs`).Scan(&runs); err != nil || runs != 0 {
				t.Fatalf("seeding called execution: runs=%d error=%v", runs, err)
			}
			// Encode/decode pins the full source selection, rather than a latest ref.
			var frozen map[string]json.RawMessage
			if json.Unmarshal(encoded, &frozen) != nil || frozen["initial_assertions"] == nil {
				t.Fatal("missing frozen selection")
			}
		})
	}
}

func TestSharedManualAssertionAndCheckAreAtomicAndReplayExactPins(t *testing.T) {
	f := newCommonTaskServiceFixture(t, checkingProgram)
	ctx := t.Context()
	content := HypothesisContent{Key: "analyst", Statement: "A manually scoped assertion", Scope: map[string]string{"conditions": "Selected actor only"}, Grounds: "Explicit analyst observation"}
	request := TaskCreationParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision, IdempotencyKey: "atomic-check", RequestDigest: contentdigest.Bytes([]byte("atomic-check")), Work: []auditcontract.PreparedSelection{{Key: "atomic-first", Workflow: "inspect", Objective: "Check the assertion", Inputs: map[string]auditcontract.Material{}}}, Assertions: map[string]HypothesisContent{"atomic-first": content}, Attention: []string{"atomic-first"}}
	result, err := f.service.CreateTasks(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	first, err := f.service.Task(ctx, "owner", f.audit.AuditID, result.Acceptance.TaskIDs[0])
	if err != nil || first.Hypothesis == nil || first.Hypothesis.Revision != 1 || !first.Attention {
		t.Fatalf("task=%+v error=%v", first, err)
	}
	request.ExpectedRevision = result.Acceptance.AuditRevision
	request.IdempotencyKey = "atomic-edit"
	request.RequestDigest = contentdigest.Bytes([]byte("atomic-edit"))
	request.Work[0].Key = "atomic-second"
	content.Statement = "A revised manually scoped assertion"
	request.Assertions = map[string]HypothesisContent{"atomic-second": content}
	request.Hypotheses = map[string]string{"atomic-second": first.Hypothesis.ID}
	request.Attention = []string{"atomic-second"}
	edited, err := f.service.CreateTasks(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	second, err := f.service.Task(ctx, "owner", f.audit.AuditID, edited.Acceptance.TaskIDs[0])
	if err != nil || second.Hypothesis.Revision != 2 || second.Hypothesis.Statement != content.Statement {
		t.Fatalf("task=%+v error=%v", second, err)
	}
	replay, err := f.service.CreateTasks(ctx, request)
	if err != nil || !replay.Replayed || replay.Acceptance.AuditRevision != edited.Acceptance.AuditRevision {
		t.Fatalf("replay=%+v error=%v", replay, err)
	}
	old, err := f.service.Task(ctx, "owner", f.audit.AuditID, first.Document.TaskID)
	if err != nil || old.Hypothesis.Revision != 1 || old.Hypothesis.Statement == content.Statement {
		t.Fatal("original pin rebound", err)
	}
	// Exhaust the persisted budget: the owner must roll back a pending revision
	// together with its rejected Task, leaving neither an orphan nor a receipt.
	if _, err := f.pool.Exec(ctx, `UPDATE audits SET task_count=max_items_total WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	request.ExpectedRevision = edited.Acceptance.AuditRevision
	request.IdempotencyKey = "budget-failed"
	request.RequestDigest = contentdigest.Bytes([]byte("budget-failed"))
	request.Work[0].Key = "atomic-third"
	request.Assertions = map[string]HypothesisContent{"atomic-third": content}
	request.Hypotheses = map[string]string{"atomic-third": first.Hypothesis.ID}
	request.Attention = nil
	if _, err := f.service.CreateTasks(ctx, request); !errors.Is(err, auditstore.ErrTaskBudgetExhausted) {
		t.Fatalf("budget rejection=%v", err)
	}
	current, err := f.service.Hypothesis(ctx, "owner", f.audit.AuditID, first.Hypothesis.ID, 2)
	if err != nil || current.Document.Revision != 2 {
		t.Fatal(err)
	}
	if _, err := f.service.Hypothesis(ctx, "owner", f.audit.AuditID, first.Hypothesis.ID, 3); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("orphan revision=%v", err)
	}
}

func TestSharedHypothesisPagesAndSparseHistoryAtRealisticSize(t *testing.T) {
	f := newCommonTaskServiceFixture(t, checkingProgram)
	ctx := t.Context()
	revision := f.audit.Revision
	var firstID string
	for start := 0; start < 1000; start += 64 {
		request := TaskCreationParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: revision, IdempotencyKey: fmt.Sprintf("history-%d", start), RequestDigest: contentdigest.Bytes([]byte(fmt.Sprintf("history-%d", start))), Assertions: map[string]HypothesisContent{}}
		for index := start; index < min(start+64, 1000); index++ {
			key := fmt.Sprintf("assertion-%04d", index)
			request.Work = append(request.Work, auditcontract.PreparedSelection{Key: key, Workflow: "inspect", Objective: "Check the selected assertion", Inputs: map[string]auditcontract.Material{}})
			request.Assertions[key] = HypothesisContent{Key: key, Statement: "An independently scoped assertion", Scope: map[string]string{"case": key}, Grounds: "Explicit analyst observation"}
		}
		accepted, err := f.service.CreateTasks(ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		revision = accepted.Acceptance.AuditRevision
		if start == 0 {
			task, err := f.service.Task(ctx, "owner", f.audit.AuditID, accepted.Acceptance.TaskIDs[0])
			if err != nil {
				t.Fatal(err)
			}
			firstID = task.Hypothesis.ID
		}
	}
	if _, err := f.pool.Exec(ctx, `UPDATE audit_tasks SET state='settled',gap_code='fixture-inspected' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	ordinal := 0
	for {
		page, err := f.service.Hypotheses(ctx, auditstore.ListHypothesesParams{OwnerID: "owner", AuditID: f.audit.AuditID, AfterOrdinal: ordinal, Limit: 200})
		if err != nil {
			t.Fatal(err)
		}
		if len(page.Items) == 0 || len(page.Items) > 200 || page.FixedDenominator != 2 {
			t.Fatalf("invalid bounded page=%+v", page)
		}
		for _, h := range page.Items {
			if seen[h.Document.ID] {
				t.Fatal("duplicate keyset member")
			}
			seen[h.Document.ID] = true
		}
		if page.NextOrdinal == nil {
			break
		}
		if *page.NextOrdinal <= ordinal {
			t.Fatal("nonadvancing continuation")
		}
		ordinal = *page.NextOrdinal
	}
	if len(seen) != 1000 {
		t.Fatalf("registry page membership=%d", len(seen))
	}
	request := TaskCreationParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: revision, IdempotencyKey: "sparse-followup", RequestDigest: contentdigest.Bytes([]byte("sparse-followup")), Work: []auditcontract.PreparedSelection{{Key: "followup", Workflow: "inspect", Objective: "Recheck one exact assertion", Inputs: map[string]auditcontract.Material{}}}, Hypotheses: map[string]string{"followup": firstID}, Attention: []string{"followup"}}
	if _, err := f.service.CreateTasks(ctx, request); err != nil {
		t.Fatal(err)
	}
	ready := auditstore.TaskReady
	page, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, State: &ready, HypothesisID: firstID, Limit: 10})
	if err != nil || len(page.Tasks) != 1 || page.Tasks[0].Ordinal != 1001 || !page.Tasks[0].Attention {
		t.Fatalf("sparse selection=%+v error=%v", page, err)
	}
	history, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, HypothesisID: firstID, Limit: 1})
	if err != nil || len(history.Tasks) != 1 || history.NextOrdinal == nil {
		t.Fatalf("history=%+v error=%v", history, err)
	}
	next, err := f.service.Tasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, HypothesisID: firstID, AfterOrdinal: *history.NextOrdinal, Limit: 1})
	if err != nil || len(next.Tasks) != 1 || next.Tasks[0].Ordinal != 1001 || next.NextOrdinal != nil {
		t.Fatalf("history continuation=%+v error=%v", next, err)
	}
}
