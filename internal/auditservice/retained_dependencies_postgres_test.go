package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestPostgresAcceptedRetainedDependencyManifestMaximumReplayAndPurge(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	catalog := loadAuditServiceProfilesWithEdits(t, nil, nil, func(files map[string]string) {
		profile := files["audit-profiles/checklist.yaml"]
		profile = strings.Replace(profile, "maxRounds: 1", "maxRounds: 3", 1)
		profile = strings.ReplaceAll(profile, "maxItemsPerRound: 10", "maxItemsPerRound: 40")
		profile = strings.ReplaceAll(profile, "maxItemsTotal: 10", "maxItemsTotal: 40")
		profile = strings.ReplaceAll(profile, "maxSubmittedRuns: 20", "maxSubmittedRuns: 80")
		var mappings, inputs, workflowInputs strings.Builder
		for i := range 30 {
			name := fmt.Sprintf("guide%02d", i)
			fmt.Fprintf(&inputs, "    %s: {required: false, mediaTypes: [application/json]}\n", name)
			fmt.Fprintf(&workflowInputs, "    %s: {required: false, mediaTypes: [application/json]}\n", name)
			fmt.Fprintf(&mappings, "        %s: {source: retained-dependency, name: baseline.%s}\n", name, name)
		}
		profile = strings.Replace(profile, "  inventory:\n", inputs.String()+"  inventory:\n", 1)
		profile = strings.Replace(profile, "        task: {source: item-package}\n", "        task: {source: item-package}\n"+mappings.String(), 1)
		files["audit-profiles/checklist.yaml"] = profile
		files["workflows/check.yaml"] = strings.Replace(files["workflows/check.yaml"], "  outputs:\n", workflowInputs.String()+"  outputs:\n", 1)
	})
	gateway, _ := catalog.LLMGateway("test-gateway@1")
	credentials := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	service, err := New(Options{Pool: pool, Profiles: &switchableProfileCatalog{snapshot: catalog, available: true}, CredentialGuard: &countingCredentialGuard{}, TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil })})
	if err != nil {
		t.Fatal(err)
	}
	const owner, project, id = "dependency-owner", "dependency-project", "dependency-audit"
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: project, OwnerID: owner, Kind: projectstore.KindProject, Name: "Dependencies", IdempotencyKey: "project", RequestDigest: serviceTestDigest("project")}); err != nil {
		t.Fatal(err)
	}
	artifactService := artifacts.NewService(artifactpostgres.NewRepository(pool))
	projectArtifacts, _ := artifactService.Project(project)
	checklist := writeChecklist(t, ctx, projectArtifacts, "dependency", "automatic")

	inputs := map[string]contracts.ArtifactRef{"checklist": checklist.Ref}
	sourcePayloads := map[string][]byte{}
	var retainedBytes int64
	for i := range 30 {
		name := fmt.Sprintf("guide%02d", i)
		payload := []byte(fmt.Sprintf(`{"guide":%d}`, i))
		written, err := projectArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "guides", Name: name}, artifacts.Payload{MediaType: "application/json", Data: payload}, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = written.Ref
		sourcePayloads[written.Ref.Key()] = payload
		retainedBytes += written.Size
	}

	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{AuditID: id, OwnerID: owner, ProjectID: project, Profile: ProfileSelector{Name: "test-checklist", Version: "1"}, Inputs: inputs, IdempotencyKey: "draft", RequestDigest: serviceTestDigest("draft")})
	if err != nil {
		t.Fatal(err)
	}
	started, err := service.Start(ctx, StartParams{OwnerID: owner, AuditID: id, ExpectedRevision: draft.Revision, IdempotencyKey: "start", RequestDigest: serviceTestDigest("start")})
	if err != nil {
		t.Fatal(err)
	}
	store := auditstoretest.NewStore(pool).WithPurgeDependencies(findingintake.DeleteAuditHoldsBatch)
	claims, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: "dependency-test", Lease: 2 * time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim: %+v %v", claims, err)
	}
	claim := claims[0]
	round := closeRoundForTest(t, ctx, store, claim, started.Round)
	document := auditFindingDocument("dependency", auditdomain.ProposedCheck{Objective: "Confirm the exact original context.", Method: "static-trace"})
	for range 16 {
		document.ProposedChecks = append(document.ProposedChecks, document.ProposedChecks[0])
	}
	seedAuditFindingProjected(t, ctx, pool, project, owner, id, "dependency", document, auditdomain.UnavailableFindingFingerprint("not_recorded"), auditdomain.UnavailableFindingAnchors("not_recorded"))
	// 2,000 unrelated bindings share one version; no dependency availability
	// read or holder acquisition may scan this containing Project population.
	unrelated, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := unrelated.Exec(ctx, `INSERT INTO artifact_bindings(scope_kind,scope_id,namespace,name,current_revision,frozen,retained_revisions,retained_bytes)
 SELECT 'project',$1,'unrelated','binding-'||n,'revision-'||n,false,1,$2 FROM generate_series(1,2000) AS n`, project, checklist.Size); err != nil {
		t.Fatal(err)
	}
	if _, err := unrelated.Exec(ctx, `INSERT INTO artifact_binding_revisions(scope_kind,scope_id,namespace,name,revision,version_id)
 SELECT 'project',$1,'unrelated','binding-'||n,'revision-'||n,source.version_id FROM generate_series(1,2000) AS n CROSS JOIN artifact_binding_revisions AS source
 WHERE source.scope_kind='project' AND source.scope_id=$1 AND source.namespace=$2 AND source.name=$3 AND source.revision=$4`, project, checklist.Ref.Namespace, checklist.Ref.Name, *checklist.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	if err := unrelated.Commit(ctx); err != nil {
		t.Fatal(err)
	}

	unrelatedRef, _ := json.Marshal(checklist.Ref)
	if _, err := pool.Exec(ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
 SELECT $1,'unrelated/'||n,$2::jsonb,$3,'application/json',$4,'{}'::jsonb FROM generate_series(1,2000) AS n`, id, unrelatedRef, func() string {
		baseline, _ := DecodeBaseline(started.Audit.BaselineSnapshot)
		return baseline.Inputs["checklist"].Digest
	}(), checklist.Size); err != nil {
		t.Fatal(err)
	}
	live, _ := store.Get(ctx, owner, id)
	params, reason, err := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if err != nil || reason != nil || len(params.Items) != 16 {
		t.Fatalf("prepare: %+v %v items=%d", reason, err, len(params.Items))
	}
	manifest := params.Items[0].Dependencies
	if manifest == nil || len(manifest.Dependencies) != 32 || len(params.Retained) != 30 {
		t.Fatalf("manifest/once-only retention: %+v links=%d", manifest, len(params.Retained))
	}
	for _, item := range params.Items {
		if item.Dependencies == nil || len(item.Dependencies.Dependencies) != 32 {
			t.Fatal("partial closure became visible")
		}
	}
	replayParams, replayReason, replayErr := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: live, Round: &round})
	if replayErr != nil || replayReason != nil || !replayParams.Manifest.Ref.SameExact(params.Manifest.Ref) {
		t.Fatalf("lost retention reply did not reconcile exact work: %v %v", replayReason, replayErr)
	}
	before := live.RetainedEvidenceBytes

	// A definite concurrent budget refusal is permanent, distinct from stale
	// state. Roll back the fixture's synthetic competing debit afterwards.
	budgetTx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := budgetTx.Exec(ctx, `UPDATE audits SET retained_evidence_bytes=max_evidence_bytes WHERE audit_id=$1`, id); err != nil {
		t.Fatal(err)
	}
	if _, _, err := auditstoretest.NewStore(budgetTx).AcceptNextRound(ctx, params); !errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		t.Fatalf("budget race was not classified permanently: %v", err)
	}
	if err := budgetTx.Rollback(ctx); err != nil {
		t.Fatal(err)
	}
	accepted, inserted, err := store.AcceptNextRound(ctx, params)
	if err != nil || !inserted {
		t.Fatalf("accept: %t %v", inserted, err)
	}
	if _, inserted, err := store.AcceptNextRound(ctx, params); err != nil || inserted {
		t.Fatalf("lost reply replay: %t %v", inserted, err)
	}
	after, _ := store.Get(ctx, owner, id)
	if after.RetainedEvidenceBytes != before+retainedBytes {
		t.Fatalf("distinct revision charged %d, want %d", after.RetainedEvidenceBytes-before, retainedBytes)
	}
	items, err := store.ListRoundItems(ctx, id, accepted.RoundID)
	if err != nil || len(items) != 16 || items[0].Dependencies == nil {
		t.Fatalf("persisted manifest: %+v %v", items, err)
	}
	encoded, _ := auditdomain.EncodeRetainedDependencies(*manifest)
	var storedBytes []byte
	if err := pool.QueryRow(ctx, `SELECT retained_dependencies::text FROM audit_items WHERE item_id=$1`, items[0].ItemID).Scan(&storedBytes); err != nil || string(storedBytes) != string(encoded) {
		t.Fatalf("stored canonical manifest bytes changed: %v", err)
	}
	stored, _ := auditdomain.EncodeRetainedDependencies(*items[0].Dependencies)
	if string(encoded) != string(stored) {
		t.Fatal("accepted manifest changed")
	}

	// One 512-descriptor slice accepted the first16 checks. The final ordinal
	// remains eligible and forms a later immutable Round with the same copies.
	closed := closeRoundForTest(t, ctx, store, claim, accepted)
	current, _ := store.Get(ctx, owner, id)
	remainingParams, remainingReason, remainingErr := service.PrepareNextRound(ctx, claim, auditstore.ReconcileSnapshot{Audit: current, Round: &closed})
	if remainingErr != nil || remainingReason != nil || len(remainingParams.Items) != 1 || remainingParams.Items[0].ProposalSources[0].ProposedCheckOrdinal != 16 {
		t.Fatalf("bounded prefix lost remaining checks: items=%d reason=%v err=%v", len(remainingParams.Items), remainingReason, remainingErr)
	}
	if _, _, err := store.AcceptNextRound(ctx, remainingParams); err != nil {
		t.Fatal(err)
	}
	after, _ = store.Get(ctx, owner, id)
	if after.RetainedEvidenceBytes != before+retainedBytes {
		t.Fatal("later shared exact revision charged twice")
	}

	changedSource := inputs["guide00"]
	versionless := changedSource.Clone()
	versionless.Revision = nil
	if _, err := projectArtifacts.Write(ctx, versionless, artifacts.Payload{MediaType: "application/json", Data: []byte(`{"changed":true}`)}, changedSource.Revision); err != nil {
		t.Fatal(err)
	}
	for _, dependency := range manifest.Dependencies {
		if dependency.Kind != "baseline" {
			continue
		}
		read, err := projectArtifacts.ReadExact(ctx, dependency.Retained)
		if err != nil || string(read.Payload.Data) != string(sourcePayloads[dependency.Source.Ref.Key()]) || dependency.Source.Ref.SameExact(dependency.Retained.Ref) {
			t.Fatalf("exact retained source: %+v %v", dependency, err)
		}
	}
	// DB-level immutable manifest and trusted-origin proof are independent of
	// Go validation and cannot be bypassed with raw mutation.
	if _, err := pool.Exec(ctx, `UPDATE audit_items SET retained_dependencies=jsonb_set(retained_dependencies::jsonb,'{owner_id}','"foreign"') WHERE item_id=$1`, items[0].ItemID); err == nil {
		t.Fatal("accepted manifest was mutable")
	}
	var holds int
	retained := params.Retained[0].Artifact
	if err := pool.QueryRow(ctx, `SELECT hold_count FROM artifact_binding_revisions WHERE scope_kind='project' AND scope_id=$1 AND namespace=$2 AND name=$3 AND revision=$4`, project, retained.Ref.Namespace, retained.Ref.Name, *retained.Ref.Revision).Scan(&holds); err != nil || holds < 2 {
		t.Fatalf("indexed lineage+manifest holds=%d %v", holds, err)
	}
	deleted, err := service.Delete(ctx, MutationParams{OwnerID: owner, AuditID: id, ExpectedRevision: after.Revision, IdempotencyKey: "delete", RequestDigest: serviceTestDigest("delete")})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ReleaseClaim(ctx, claim); err != nil {
		t.Fatal(err)
	}
	deleteClaim := claimAuditForPurge(t, ctx, store, deleted.Audit.AuditID)
	if _, err := store.SettleUndispatched(ctx, deleteClaim, auditstore.MaxReconcileRows); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.ReleaseDispatchHold(ctx, deleteClaim); err != nil {
		t.Fatal(err)
	}
	deleting, err := store.Get(ctx, owner, id)
	if err != nil {
		t.Fatal(err)
	}
	if deleting.State == auditstore.AuditCancelling {
		if _, err := store.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{Claim: deleteClaim, ExpectedRevision: deleting.Revision, ExpectedState: auditstore.AuditCancelling, TargetState: auditstore.AuditDeleting, Reason: deleting.StopReason}); err != nil {
			t.Fatal(err)
		}
	}

	// Test instrumentation records actual domain deletes and exact-revision
	// counter updates. It does not infer work from primary RowsAffected alone.
	if _, err := pool.Exec(ctx, `CREATE TABLE dependency_purge_work(kind text,statement text);
 CREATE FUNCTION record_dependency_purge_work() RETURNS trigger LANGUAGE plpgsql AS $$
 BEGIN INSERT INTO dependency_purge_work VALUES(TG_TABLE_NAME,current_query()); RETURN NULL; END; $$;
 CREATE TRIGGER count_dependency_hold_work AFTER UPDATE OF hold_count ON artifact_binding_revisions
 FOR EACH ROW WHEN (OLD.hold_count IS DISTINCT FROM NEW.hold_count) EXECUTE FUNCTION record_dependency_purge_work()`); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"audit_items", "audit_execution_items", "audit_executions", "audit_rounds", "audit_artifact_links", "audit_review_requests", "audit_review_decisions", "audit_report_pages", "audit_events", "audits"} {
		if _, err := pool.Exec(ctx, `CREATE TRIGGER count_dependency_domain_work AFTER DELETE ON `+table+` FOR EACH ROW EXECUTE FUNCTION record_dependency_purge_work()`); err != nil {
			t.Fatal(err)
		}
	}
	sawHoldWork := false
	for pass := 0; pass < 30; pass++ {
		if _, err := pool.Exec(ctx, `TRUNCATE dependency_purge_work`); err != nil {
			t.Fatal(err)
		}
		if err := store.PurgeClaimed(ctx, deleteClaim, auditdomain.ArtifactNamespace(id)); err != nil {
			t.Fatal(err)
		}
		var domainWork, holdWork int
		if err := pool.QueryRow(ctx, `SELECT count(*) FILTER(WHERE kind<>'artifact_binding_revisions'),count(*) FILTER(WHERE kind='artifact_binding_revisions' AND statement LIKE '%weighted AS MATERIALIZED%') FROM dependency_purge_work`).Scan(&domainWork, &holdWork); err != nil {
			t.Fatal(err)
		}
		if domainWork+holdWork > persistencepostgres.CleanupBatchRows {
			t.Fatalf("purge pass=%d mutated %d domain rows + %d exact counters, bound=%d", pass, domainWork, holdWork, persistencepostgres.CleanupBatchRows)
		}
		sawHoldWork = sawHoldWork || holdWork > 0
		var exists bool
		if err := pool.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audits WHERE audit_id=$1)`, id).Scan(&exists); err != nil {
			t.Fatal(err)
		}
		if !exists {
			break
		}
		if pass == 29 {
			t.Fatal("purge did not finish within bounded passes")
		}
	}
	if !sawHoldWork {
		t.Fatal("populated purge did not exercise exact dependency counter updates")
	}
	var remaining int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$1 AND namespace=$2`, project, auditdomain.ArtifactNamespace(id)).Scan(&remaining); err != nil || remaining != 0 {
		t.Fatalf("orphan dependency bindings=%d %v", remaining, err)
	}
}
