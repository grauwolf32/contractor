package auditservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func newTimeControlAudit(t *testing.T, seconds int) (context.Context, *pgxpool.Pool, *Service, StartedAudit, *time.Time) {
	t.Helper()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	pool := isolatedAuditServicePool(t, ctx, url)
	profiles := loadAuditServiceProfiles(t)
	gateway, err := profiles.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	service, err := New(Options{Storage: artifactpgtest.NewStorage(), Pool: pool, Profiles: profiles, CredentialGuard: &countingCredentialGuard{}, Now: func() time.Time { return now }, TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) {
		return &switchableCredentialLookup{available: true, gateway: gateway.Ref}, nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project-time", OwnerID: "owner-time", Kind: projectstore.KindProject, Name: "Time controls", IdempotencyKey: "project-time", RequestDigest: serviceTestDigest("project-time")})
	if err != nil {
		t.Fatal(err)
	}
	scoped, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	items := []map[string]any{}
	for _, key := range []string{"accepted", "remaining", "cancelled", "manual"} {
		policy := "automatic"
		if key == "manual" {
			policy = "manual"
		}
		items = append(items, map[string]any{"key": key, "version": "1", "statement": "Verify " + key, "applicability": "always", "allowed_methods": []string{"static"}, "required_evidence": []string{}, "review_policy": policy})
	}
	data, _ := json.Marshal(map[string]any{"schema": "contractor.audit.checklist.v1", "items": items})
	input, err := scoped.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "checks"}, artifacts.Payload{MediaType: "application/json", Data: data}, nil)
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{AuditID: "audit-time", OwnerID: project.OwnerID, ProjectID: project.ProjectID, Profile: ProfileSelector{Name: "test-checklist", Version: "1"}, Inputs: map[string]contracts.ArtifactRef{"checklist": input.Ref}, IdempotencyKey: "create-time", RequestDigest: serviceTestDigest("create-time")})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Start(ctx, StartParams{OwnerID: draft.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start-time", RequestDigest: serviceTestDigest("start-time"), DeadlineSeconds: &seconds})
	if err != nil {
		t.Fatal(err)
	}
	return ctx, pool, service, started, &now
}

func TestAuditPausePreservesRemainingTimeAndResumeCanDisableDeadline(t *testing.T) {
	ctx, _, service, started, now := newTimeControlAudit(t, 7200)
	if started.Audit.DeadlineAt == nil || !started.Audit.DeadlineAt.Equal(now.Add(2*time.Hour)) {
		t.Fatal("custom start deadline was not applied")
	}
	paused, err := service.Pause(ctx, MutationParams{OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, ExpectedRevision: started.Audit.Revision, IdempotencyKey: "pause-time", RequestDigest: serviceTestDigest("pause-time")})
	if err != nil {
		t.Fatal(err)
	}
	if paused.Audit.PausedAt == nil {
		t.Fatal("pause timestamp missing")
	}
	remaining := paused.Audit.DeadlineAt.Sub(*paused.Audit.PausedAt)
	*now = paused.Audit.PausedAt.Add(48 * time.Hour)
	params := MutationParams{OwnerID: paused.Audit.OwnerID, AuditID: paused.Audit.AuditID, ExpectedRevision: paused.Audit.Revision, IdempotencyKey: "resume-time", RequestDigest: serviceTestDigest("resume-time")}
	resumed, err := service.Resume(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Audit.PausedAt != nil || resumed.Audit.DeadlineAt == nil || !resumed.Audit.DeadlineAt.Equal(now.Add(remaining)) {
		t.Fatal("paused time consumed the allowance")
	}
	replay, err := service.Resume(ctx, params)
	if err != nil || !replay.Replayed || replay.Audit.Revision != resumed.Audit.Revision {
		t.Fatalf("resume replay: %v", err)
	}
	params.IdempotencyKey = "stale-resume"
	params.RequestDigest = serviceTestDigest("stale-resume")
	if _, err := service.Resume(ctx, params); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("stale resume: %v", err)
	}
	paused, err = service.Pause(ctx, MutationParams{OwnerID: resumed.Audit.OwnerID, AuditID: resumed.Audit.AuditID, ExpectedRevision: resumed.Audit.Revision, IdempotencyKey: "pause-again", RequestDigest: serviceTestDigest("pause-again")})
	if err != nil {
		t.Fatal(err)
	}
	zero := 0
	params = MutationParams{OwnerID: paused.Audit.OwnerID, AuditID: paused.Audit.AuditID, ExpectedRevision: paused.Audit.Revision, IdempotencyKey: "resume-unlimited", RequestDigest: serviceTestDigest("resume-unlimited"), DeadlineSeconds: &zero}
	resumed, err = service.Resume(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Audit.DeadlineAt != nil {
		t.Fatal("resume failed to disable deadline")
	}
}

func TestAuditStartWithoutDeadline(t *testing.T) {
	_, _, _, started, _ := newTimeControlAudit(t, 0)
	if started.Audit.DeadlineAt != nil {
		t.Fatal("unlimited start has a deadline")
	}
}

func TestAuditRejectsTerminalDeadlineResumeWithoutChangingRetainedState(t *testing.T) {
	for _, state := range []auditstore.AuditState{auditstore.AuditCompleted, auditstore.AuditFailed} {
		t.Run(string(state), func(t *testing.T) {
			ctx, pool, service, started, _ := newTimeControlAudit(t, 7200)
			if _, err := pool.Exec(ctx, `UPDATE audits SET state=$1,dispatch_state='closed',hold_state='released',deadline_at=clock_timestamp()-interval '1 hour',finished_at=clock_timestamp(),stop_reason_code='deadline_exhausted',stop_reason_message='Legacy deadline' WHERE audit_id='audit-time'`, state); err != nil {
				t.Fatal(err)
			}
			// Keep terminal common work and exact baseline materials unchanged.
			if _, err := pool.Exec(ctx, `UPDATE audit_tasks SET state='settled',gap_code='audit-closed-before-dispatch' WHERE audit_id='audit-time' AND state<>'settled'`); err != nil {
				t.Fatal(err)
			}
			before := auditTimeControlSnapshot(t, ctx, pool)
			zero, hour := 0, 3600
			for _, override := range []*int{nil, &zero, &hour} {
				params := MutationParams{OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, ExpectedRevision: started.Audit.Revision, IdempotencyKey: "continue-time", RequestDigest: serviceTestDigest("continue-time"), DeadlineSeconds: override}
				for range 2 {
					if _, err := service.Resume(ctx, params); !errors.Is(err, auditstore.ErrPrecondition) {
						t.Fatalf("terminal Resume = %v, want precondition failure", err)
					}
				}
			}
			if after := auditTimeControlSnapshot(t, ctx, pool); !bytes.Equal(before, after) {
				t.Fatal("terminal Resume changed retained authority, reports or Tasks")
			}
		})
	}
}

func auditTimeControlSnapshot(t *testing.T, ctx context.Context, pool *pgxpool.Pool) []byte {
	t.Helper()
	var data []byte
	if err := pool.QueryRow(ctx, `SELECT jsonb_build_object(
'audit', (SELECT to_jsonb(a) FROM audits a WHERE audit_id='audit-time'),
'tasks', (SELECT jsonb_agg(to_jsonb(i) ORDER BY task_id) FROM audit_tasks i WHERE audit_id='audit-time'),
'links', (SELECT jsonb_agg(to_jsonb(l) ORDER BY logical_key) FROM audit_artifact_links l WHERE audit_id='audit-time'),
'events', (SELECT jsonb_agg(to_jsonb(e) ORDER BY sequence_number) FROM audit_events e WHERE audit_id='audit-time'),
'receipts', (SELECT jsonb_agg(to_jsonb(i) ORDER BY idempotency_key) FROM mutation_receipts i WHERE audit_id='audit-time')
)`).Scan(&data); err != nil {
		t.Fatal(err)
	}
	return data
}
