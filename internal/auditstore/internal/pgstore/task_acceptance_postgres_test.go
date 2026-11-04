package pgstore

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5/pgxpool"
)

type commonTaskFixture struct {
	pool     *pgxpool.Pool
	owners   *OwnerStore
	reads    *QueryStore
	audit    auditstore.Audit
	pinned   auditprofile.PinnedProgram
	material auditcontract.Material
	service  *artifacts.Service
	source   contracts.ArtifactRef
}

func newCommonTaskFixture(t *testing.T, connections int32, tighten ...func(*auditstore.Limits)) commonTaskFixture {
	t.Helper()
	ctx := t.Context()
	pool := postgrestest.NewPool(t, ctx, "", postgrestest.Options{Configure: func(cfg *pgxpool.Config) { cfg.MaxConns = connections }})
	return commonTaskFixtureOnPool(t, pool, "", tighten...)
}
func commonTaskFixtureOnPool(t *testing.T, pool *pgxpool.Pool, name string, tighten ...func(*auditstore.Limits)) commonTaskFixture {
	return setupCommonTaskFixture(t, pool, name, true, tighten...)
}
func setupCommonTaskFixture(t *testing.T, pool *pgxpool.Pool, name string, start bool, tighten ...func(*auditstore.Limits)) commonTaskFixture {
	t.Helper()
	ctx := t.Context()
	auditID, ownerID, projectID := "tasks-audit", "owner", "tasks-project"
	if name != "" {
		auditID = "audit-" + name
		ownerID = "owner-" + name
		projectID = "project-" + name
	}
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	root := configtest.CopyWithPolicies(t, "../../../config/testdata/valid")
	if err := os.CopyFS(root, os.DirFS("../../../config/testdata/audit-common-catalog")); err != nil {
		t.Fatal(err)
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := catalog.AuditProfile("common-tasks@1")
	if err != nil {
		t.Fatal(err)
	}
	profile, _ := json.Marshal(pinned)
	project := projectstore.NewPostgresStore(pool)
	if _, _, err := project.Create(ctx, projectstore.CreateParams{ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject, Name: "Common task intake", IdempotencyKey: "project", RequestDigest: contentdigest.Bytes([]byte("project"))}); err != nil {
		t.Fatal(err)
	}
	limits := pinned.Program.Limits
	owners, reads := NewOwnerStore(pool, artifactpgtest.NewStorage()), NewQueryStore(pool, artifactpgtest.NewStorage())
	effective := auditstore.Limits{MaxRounds: limits.MaxIterations, BatchSize: limits.BatchSize, MaxItemsPerRound: limits.MaxTasksPerIteration, MaxItemsTotal: limits.MaxTasks, MaxSubmittedRuns: limits.MaxSubmittedRuns, MaxItemRunAttempts: limits.MaxAttempts, MaxEvidenceBytes: limits.MaxEvidenceBytes}
	for _, change := range tighten {
		change(&effective)
	}
	audit, _, err := owners.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile: auditstore.ProfileIdentity{Name: pinned.Ref.Name, Version: pinned.Ref.Version, Digest: pinned.Ref.Digest}, ProfileSnapshot: profile, InputSelection: json.RawMessage(`{}`),
		Limits:         effective,
		IdempotencyKey: "draft", RequestDigest: contentdigest.Bytes([]byte("draft")),
	})
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool))
	projectArtifacts, _ := service.Project(audit.ProjectID)
	source, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("selected source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	projectScope, _ := artifacts.ProjectScope(audit.ProjectID)
	auditScope, _ := artifacts.AuditScope(audit.AuditID)
	retained, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditInput, Source: projectScope, SourceRef: source.Ref, Target: auditScope, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "selected-source"}, Frozen: true})
	if err != nil {
		t.Fatal(err)
	}
	metadata, err := artifactpgtest.NewStorage().Repository(pool).Metadata(ctx, auditScope, retained.TargetRef)
	if err != nil || !metadata.Frozen {
		t.Fatalf("retained material: %v", err)
	}
	material := auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: metadata.Ref, Digest: metadata.Digest, MediaType: metadata.MediaType, SizeBytes: metadata.Size}
	if start {
		baseline := commonTaskBaseline(source.Ref, metadata.Digest, metadata.MediaType, metadata.Size)
		audit, _, err = owners.Activate(ctx, auditstore.ActivateParams{OwnerID: ownerID, AuditID: auditID, ExpectedRevision: audit.Revision, BaselineSnapshot: baseline, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start", RequestDigest: contentdigest.Bytes([]byte("start"))})
		if err != nil {
			t.Fatal(err)
		}
	}
	return commonTaskFixture{pool: pool, owners: owners, reads: reads, audit: audit, pinned: pinned, service: service, material: material, source: source.Ref}
}

func (f commonTaskFixture) acceptance(count int, key string) auditstore.AcceptTasksParams {
	tasks := make([]auditstore.AcceptedTask, 0, count)
	for index := range count {
		id := fmt.Sprintf("task-%03d", index)
		tasks = append(tasks, auditstore.AcceptedTask{Document: auditcontract.Task{Schema: auditcontract.TaskSchema, AuditID: f.audit.AuditID, TaskID: id, Workflow: "inspect", Inputs: map[string]auditcontract.Material{"source": f.material}, Parameters: map[string]string{}, Objective: "Inspect the selected source"}, Source: auditstore.TaskSource{Kind: "manual", Key: id}})
	}
	digest, _ := contentdigest.CanonicalJSON(tasks)
	return auditstore.AcceptTasksParams{OwnerID: f.audit.OwnerID, AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision, IdempotencyKey: key, RequestDigest: digest, Tasks: tasks}
}

func TestSharedPostgresCommonTaskAcceptanceReplayAndPause(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	ctx := t.Context()
	paused, _, err := f.owners.Transition(ctx, auditstore.TransitionParams{OwnerID: "owner", AuditID: f.audit.AuditID, ExpectedRevision: f.audit.Revision, ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditPaused, IdempotencyKey: "pause", RequestDigest: contentdigest.Bytes([]byte("pause"))})
	if err != nil {
		t.Fatal(err)
	}
	f.audit = paused
	p := f.acceptance(2, "accept")
	first, replayed, err := f.owners.AcceptTasks(ctx, p)
	if err != nil || replayed || first.CreatedCount != 2 {
		t.Fatalf("accepted: %+v replay=%v error=%v", first, replayed, err)
	}
	second, replayed, err := f.owners.AcceptTasks(ctx, p)
	if err != nil || !replayed || fmt.Sprint(first) != fmt.Sprint(second) {
		t.Fatalf("replay: %+v replay=%v error=%v", second, replayed, err)
	}
	page, err := f.reads.ListTasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 1})
	if err != nil || len(page.Tasks) != 1 || page.NextOrdinal == nil || page.Tasks[0].State != auditstore.TaskReady {
		t.Fatalf("first page: %+v %v", page, err)
	}
	next, err := f.reads.ListTasks(ctx, auditstore.ListTasksParams{OwnerID: "owner", AuditID: f.audit.AuditID, Limit: 1, AfterOrdinal: *page.NextOrdinal})
	if err != nil || len(next.Tasks) != 1 || next.NextOrdinal != nil || next.Tasks[0].Document.TaskID == page.Tasks[0].Document.TaskID {
		t.Fatalf("second page: %+v %v", next, err)
	}
	audit, err := f.reads.Get(ctx, "owner", f.audit.AuditID)
	if err != nil || audit.State != auditstore.AuditPaused || audit.TaskCount != 2 || audit.SubmittedRunCount != 0 {
		t.Fatalf("paused admission changed Run/control state: %+v %v", audit, err)
	}
}

func TestSharedPostgresCommonTaskAcceptanceConcurrentKey(t *testing.T) {
	f := newCommonTaskFixture(t, 2)
	p := f.acceptance(1, "concurrent")
	var wait sync.WaitGroup
	results := make([]auditstore.TaskAcceptance, 2)
	failures := make([]error, 2)
	for index := range results {
		wait.Go(func() { results[index], _, failures[index] = f.owners.AcceptTasks(t.Context(), p) })
	}
	wait.Wait()
	for index, err := range failures {
		if err != nil || len(results[index].TaskIDs) != 1 || results[index].TaskIDs[0] != results[0].TaskIDs[0] {
			t.Fatalf("concurrent accept %d: %+v %v", index, results[index], err)
		}
	}
	var tasks int
	if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM audit_tasks WHERE audit_id=$1`, f.audit.AuditID).Scan(&tasks); err != nil || tasks != 1 {
		t.Fatalf("duplicate task: %d %v", tasks, err)
	}
}

func TestSharedPostgresCommonTaskAcceptanceRejectsForeignOrChangedContent(t *testing.T) {
	for _, change := range []string{"scope", "metadata", "binding", "context", "source-collision", "cancelling"} {
		t.Run(change, func(t *testing.T) {
			f := newCommonTaskFixture(t, 1)
			p := f.acceptance(1, change)
			switch change {
			case "scope":
				p.Tasks[0].Document.AuditID = "other-audit"
			case "metadata":
				material := p.Tasks[0].Document.Inputs["source"]
				material.SizeBytes++
				p.Tasks[0].Document.Inputs["source"] = material
			case "binding":
				p.Tasks[0].Document.Workflow = "foreign"
			case "context":
				p.Tasks[0].Document.Inputs["context"] = f.material
			case "source-collision":
				first, _, err := f.owners.AcceptTasks(t.Context(), p)
				if err != nil {
					t.Fatal(err)
				}
				p.ExpectedRevision = first.AuditRevision
				p.IdempotencyKey += "-changed"
				p.RequestDigest = contentdigest.Bytes([]byte("changed"))
				p.Tasks[0].Document.Objective = "Changed work under the same source identity"
			case "cancelling":
				if _, err := f.pool.Exec(t.Context(), `UPDATE audits SET state='cancelling' WHERE audit_id=$1`, f.audit.AuditID); err != nil {
					t.Fatal(err)
				}
			}
			if _, _, err := f.owners.AcceptTasks(t.Context(), p); err == nil {
				t.Fatal("invalid work was accepted")
			}
			var count int
			if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM audit_tasks WHERE audit_id=$1`, f.audit.AuditID).Scan(&count); err != nil {
				t.Fatal(err)
			}
			want := 0
			if change == "source-collision" {
				want = 1
			}
			if count != want {
				t.Fatalf("partial task write: %d", count)
			}
		})
	}
}

func TestSharedPostgresCommonTaskInputsStayExactAcrossProjectEdits(t *testing.T) {
	f := newCommonTaskFixture(t, 1)
	p := f.acceptance(1, "retained")
	if _, _, err := f.owners.AcceptTasks(t.Context(), p); err != nil {
		t.Fatal(err)
	}
	policy, err := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.HistoryBytesFloor)
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	historyService := artifacts.NewService(artifactpgtest.NewConfigured(artifactpgtest.NewRuntime(), policy).Repository(f.pool))
	project, _ := historyService.Project(f.audit.ProjectID)
	current := f.source
	var intermediate contracts.ArtifactRef
	for index, value := range []string{"source revision two", "source revision three"} {
		written, err := project.Write(ctx, contracts.ArtifactRef{Namespace: current.Namespace, Name: current.Name}, artifacts.Payload{MediaType: "text/plain", Data: []byte(value)}, current.Revision)
		if err != nil {
			t.Fatal(err)
		}
		current = written.Ref
		if index == 0 {
			intermediate = written.Ref
		}
	}
	// Ordinary lineage protects the exact selected source. Retention may prune
	// the intermediate revision while accepted Audit bytes stay immutable.
	if _, err := project.Read(ctx, intermediate); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("intermediate history was not pruned: %v", err)
	}
	owned, _ := f.service.Scoped(mustCommonAuditScope(t, f.audit.AuditID))
	read, err := owned.Read(ctx, f.material.Ref)
	if err != nil || string(read.Payload.Data) != "selected source" {
		t.Fatalf("accepted bytes changed with source: %v", err)
	}
	task, err := f.reads.GetTask(ctx, "owner", f.audit.AuditID, p.Tasks[0].Document.TaskID)
	if err != nil {
		t.Fatal(err)
	}
	wanted, _ := strictjson.Canonical(f.material)
	actual, _ := strictjson.Canonical(task.Document.Inputs["source"])
	if string(wanted) != string(actual) {
		t.Fatal("accepted input pin changed")
	}
}

func mustCommonAuditScope(t *testing.T, auditID string) artifacts.Scope {
	t.Helper()
	scope, err := artifacts.AuditScope(auditID)
	if err != nil {
		t.Fatal(err)
	}
	return scope
}

func TestSharedPostgresCommonTaskMutationBounds(t *testing.T) {
	for _, scenario := range []string{"128-records", "129-records", "metadata-ceiling", "metadata-overflow"} {
		t.Run(scenario, func(t *testing.T) {
			f := newCommonTaskFixture(t, 1)
			count := auditcontract.MaxMutationRecords
			if scenario == "129-records" {
				count++
			}
			p := f.acceptance(count, scenario)
			if strings.HasPrefix(scenario, "metadata-") {
				for index := range p.Tasks {
					p.Tasks[index].Document.Objective = strings.Repeat("x", 7000)
				}
				selected, err := validateAcceptedTasks(p)
				if err != nil {
					t.Fatal(err)
				}
				for index := range selected {
					selected[index].Ordinal, selected[index].Iteration = index+1, 1
				}
				encoded, _ := strictjson.Canonical(selected)
				remaining := auditcontract.MaxMutationBytes - len(encoded)
				if remaining <= 0 {
					t.Fatal("metadata fixture does not fit")
				}
				for index := range p.Tasks {
					add := min(remaining, 8192-len(p.Tasks[index].Document.Objective))
					p.Tasks[index].Document.Objective += strings.Repeat("x", add)
					remaining -= add
				}
				if remaining != 0 {
					t.Fatal("metadata ceiling was not reached")
				}
				if scenario == "metadata-overflow" {
					p.Tasks[len(p.Tasks)-1].Document.Objective += "x"
				}
			}
			_, _, err := f.owners.AcceptTasks(t.Context(), p)
			invalid := scenario == "129-records" || scenario == "metadata-overflow"
			if invalid != (err != nil) {
				t.Fatalf("invalid=%v error=%v", invalid, err)
			}
			var stored int
			if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM audit_tasks WHERE audit_id=$1`, f.audit.AuditID).Scan(&stored); err != nil {
				t.Fatal(err)
			}
			if invalid && stored != 0 || !invalid && stored != count {
				t.Fatalf("partial mutation: %d", stored)
			}
		})
	}
}

func TestSharedPostgresCommonTaskClaimAndProjectFences(t *testing.T) {
	for _, scenario := range []string{"claim-lost", "project-deleting"} {
		t.Run(scenario, func(t *testing.T) {
			f := newCommonTaskFixture(t, 1)
			p := f.acceptance(1, scenario)
			if scenario == "claim-lost" {
				claims := NewClaimStore(f.pool, artifactpgtest.NewStorage())
				claimed, err := claims.Claim(t.Context(), auditstore.ClaimParams{HolderID: "holder", Lease: time.Minute, Limit: 1})
				if err != nil || len(claimed) != 1 {
					t.Fatalf("claim: %+v %v", claimed, err)
				}
				if err := claims.ReleaseClaim(t.Context(), claimed[0]); err != nil {
					t.Fatal(err)
				}
				p.Claim, p.ExpectedRevision = &claimed[0], 0
			} else if _, _, err := projectstore.NewPostgresStore(f.pool).BeginDeletion(t.Context(), projectstore.BeginDeletionParams{OwnerID: "owner", ProjectID: f.audit.ProjectID, ExpectedRevision: 1}); err != nil {
				t.Fatal(err)
			}
			if _, _, err := f.owners.AcceptTasks(t.Context(), p); err == nil {
				t.Fatal("closed authority admitted tasks")
			}
			var count int
			if err := f.pool.QueryRow(t.Context(), `SELECT count(*) FROM audit_tasks WHERE audit_id=$1`, f.audit.AuditID).Scan(&count); err != nil || count != 0 {
				t.Fatalf("write after closed fence: %d %v", count, err)
			}
		})
	}
}

func TestSharedPostgresProjectToAuditInputForkFences(t *testing.T) {
	for _, scenario := range []string{"unfrozen", "latest", "cancelled", "foreign-project", "project-deleting"} {
		t.Run(scenario, func(t *testing.T) {
			f := newCommonTaskFixture(t, 1)
			sourceScope, _ := artifacts.ProjectScope(f.audit.ProjectID)
			request := artifacts.Fork{Kind: artifacts.LineageAuditInput, Source: sourceScope, SourceRef: f.source, Target: mustCommonAuditScope(t, f.audit.AuditID), TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "refused"}, Frozen: true}
			switch scenario {
			case "unfrozen":
				request.Frozen = false
			case "latest":
				request.SourceRef.Revision = nil
			case "cancelled":
				if _, err := f.pool.Exec(t.Context(), `UPDATE audits SET state='cancelled',finished_at=clock_timestamp() WHERE audit_id=$1`, f.audit.AuditID); err != nil {
					t.Fatal(err)
				}
			case "foreign-project":
				if _, _, err := projectstore.NewPostgresStore(f.pool).Create(t.Context(), projectstore.CreateParams{ProjectID: "foreign-project", OwnerID: "other", Kind: projectstore.KindProject, Name: "Other", IdempotencyKey: "foreign", RequestDigest: contentdigest.Bytes([]byte("foreign"))}); err != nil {
					t.Fatal(err)
				}
				foreign, _ := f.service.Project("foreign-project")
				written, err := foreign.Write(t.Context(), contracts.ArtifactRef{Namespace: "inputs", Name: "source"}, artifacts.Payload{MediaType: "text/plain", Data: []byte("foreign")}, nil)
				if err != nil {
					t.Fatal(err)
				}
				request.Source, _ = artifacts.ProjectScope("foreign-project")
				request.SourceRef = written.Ref
			case "project-deleting":
				if _, _, err := projectstore.NewPostgresStore(f.pool).BeginDeletion(t.Context(), projectstore.BeginDeletionParams{OwnerID: "owner", ProjectID: f.audit.ProjectID, ExpectedRevision: 1}); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := f.service.Fork(t.Context(), request); err == nil {
				t.Fatal("invalid Project-to-Audit input fork was admitted")
			}
			if _, err := artifactpgtest.NewStorage().Repository(f.pool).Metadata(t.Context(), request.Target, request.TargetRef); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("partial input fork: %v", err)
			}
		})
	}
}

func TestSharedPostgresCommonTaskAcceptanceUsesTighterEffectiveLimits(t *testing.T) {
	f := newCommonTaskFixture(t, 1, func(limits *auditstore.Limits) {
		limits.MaxItemsTotal = 1
		limits.MaxItemsPerRound = 1
		limits.BatchSize = 1
		limits.MaxSubmittedRuns = 1
		limits.MaxItemRunAttempts = 1
		limits.MaxEvidenceBytes = 1 << 20
	})
	tooMany := f.acceptance(2, "too-many")
	if _, _, err := f.owners.AcceptTasks(t.Context(), tooMany); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("effective task ceiling: %v", err)
	}
	accepted, _, err := f.owners.AcceptTasks(t.Context(), f.acceptance(1, "within-cap"))
	if err != nil || accepted.CreatedCount != 1 {
		t.Fatalf("valid tighter pin rejected: %+v %v", accepted, err)
	}
	audit, err := f.reads.Get(t.Context(), "owner", f.audit.AuditID)
	if err != nil || audit.TaskCount != 1 || audit.Limits.MaxItemsTotal != 1 {
		t.Fatalf("effective limits widened: %+v %v", audit, err)
	}
}

func commonTaskBaseline(ref contracts.ArtifactRef, digest, media string, size int64) json.RawMessage {
	data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.baseline.v1", "inputs": map[string]auditstore.ExactArtifact{"source": {Ref: ref, Digest: digest, MediaType: media, SizeBytes: size}}, "skills": []any{}})
	return data
}

func commonTaskBaselineInputs(material auditcontract.Material) json.RawMessage {
	data, _ := json.Marshal(map[string]auditstore.ExactArtifact{"source": {Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}})
	return data
}

func TestTaskAcceptanceReceiptDistinguishesEmptyFromNull(t *testing.T) {
	for _, value := range []string{`{"task_ids":null,"created_count":0,"audit_revision":1}`, `{"created_count":0,"audit_revision":1}`} {
		if _, err := decodeTaskAcceptance([]byte(value)); err == nil {
			t.Fatal("missing/null membership accepted as an empty producer decision")
		}
	}
	if _, err := decodeTaskAcceptance([]byte(`{"task_ids":[],"created_count":0,"audit_revision":1}`)); err != nil {
		t.Fatal(err)
	}
}
