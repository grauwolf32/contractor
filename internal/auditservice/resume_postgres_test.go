package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
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
	service, err := New(Options{Pool: pool, Profiles: profiles, CredentialGuard: &countingCredentialGuard{}, Now: func() time.Time { return now }, TransactionLLMCredentials: runtimeconfig.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (config.CredentialLookup, error) {
		return &switchableCredentialLookup{available: true, gateway: gateway.Ref}, nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "project-time", OwnerID: "owner-time", Kind: projectstore.KindProject, Name: "Time controls", IdempotencyKey: "project-time", RequestDigest: serviceTestDigest("project-time")})
	if err != nil {
		t.Fatal(err)
	}
	scoped, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
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

func TestAuditContinuesLegacyDeadlineClosureWithoutChangingAcceptedResults(t *testing.T) {
	ctx, pool, service, started, _ := newTimeControlAudit(t, 7200)
	// A legacy closure has settled undispatched work and retained a partial report.
	for _, sql := range []string{
		`UPDATE audits SET state='completed',dispatch_state='closed',hold_state='released',deadline_at=clock_timestamp()-interval '1 hour',finished_at=clock_timestamp(),stop_reason_code='deadline_exhausted',stop_reason_message='Legacy deadline' WHERE audit_id='audit-time'`,
		`UPDATE audit_rounds SET state='closed' WHERE audit_id='audit-time'`,
		`UPDATE audit_items SET state='settled',final_disposition=CASE WHEN item_key='accepted' THEN 'accepted-result' WHEN item_key='cancelled' THEN 'execution-cancelled' ELSE 'excluded' END,accepted_result_ref=CASE WHEN item_key='accepted' THEN task_ref END,accepted_result_digest=CASE WHEN item_key='accepted' THEN task_digest END WHERE audit_id='audit-time'`,
		`UPDATE audit_coverage_rows SET gaps='["audit-closed-before-dispatch"]'::jsonb WHERE audit_id='audit-time' AND item_key IN ('remaining','manual')`,
		`UPDATE audit_review_requests SET expires_at=clock_timestamp()-interval '1 hour' WHERE audit_id='audit-time'`,
		`INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance,display_ref) SELECT audit_id,'report/machine',task_ref,task_digest,'application/json',1,'{}'::jsonb,'Historical report' FROM audit_items WHERE audit_id='audit-time' AND item_key='accepted'`,
	} {
		if _, err := pool.Exec(ctx, sql); err != nil {
			t.Fatal(err)
		}
	}
	var acceptedBefore, acceptedAfter string
	if err := pool.QueryRow(ctx, `SELECT row_to_json(i)::text FROM audit_items i WHERE audit_id='audit-time' AND item_key='accepted'`).Scan(&acceptedBefore); err != nil {
		t.Fatal(err)
	}
	zero := 0
	params := MutationParams{OwnerID: started.Audit.OwnerID, AuditID: started.Audit.AuditID, ExpectedRevision: started.Audit.Revision, IdempotencyKey: "continue-time", RequestDigest: serviceTestDigest("continue-time"), DeadlineSeconds: &zero}
	resumed, err := service.Resume(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if resumed.Audit.State != auditstore.AuditActive || resumed.Audit.DeadlineAt != nil || resumed.Audit.Hold != auditstore.HoldHeld || resumed.Audit.ContinuationCount != 1 || resumed.Audit.StopReason != nil {
		t.Fatal("continuation state is wrong")
	}
	if err := pool.QueryRow(ctx, `SELECT row_to_json(i)::text FROM audit_items i WHERE audit_id='audit-time' AND item_key='accepted'`).Scan(&acceptedAfter); err != nil {
		t.Fatal(err)
	}
	if acceptedBefore != acceptedAfter {
		t.Fatal("accepted result changed")
	}
	var ready, awaiting, history, pending, expired int
	for _, check := range []struct {
		sql  string
		dest *int
	}{
		{`SELECT count(*) FROM audit_items WHERE audit_id='audit-time' AND state='ready'`, &ready},
		{`SELECT count(*) FROM audit_items WHERE audit_id='audit-time' AND state='awaiting_review'`, &awaiting},
		{`SELECT count(*) FROM audit_artifact_links WHERE audit_id='audit-time' AND logical_key LIKE 'report/history/%'`, &history},
		{`SELECT count(*) FROM audit_review_requests WHERE audit_id='audit-time' AND state='pending' AND expires_at IS NULL`, &pending},
		{`SELECT count(*) FROM audit_review_requests WHERE audit_id='audit-time' AND state='expired'`, &expired},
	} {
		if err := pool.QueryRow(ctx, check.sql).Scan(check.dest); err != nil {
			t.Fatal(err)
		}
	}
	if ready != 2 || awaiting != 1 || history != 1 || pending != 1 || expired != 1 {
		t.Fatalf("ready=%d awaiting=%d history=%d pending=%d expired=%d", ready, awaiting, history, pending, expired)
	}
	replay, err := service.Resume(ctx, params)
	if err != nil || !replay.Replayed || replay.Audit.ContinuationCount != 1 {
		t.Fatalf("continuation replay: %v", err)
	}
}
