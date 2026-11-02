package public

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestSharedPostgresAuditDetailReadsOneCommitWhileTheAuditChanges commits
// prepare attempts and their collection between the statements that read
// the detail of a preparing Audit, for GET /v1/audits/{auditId} and for a
// pause response, and validates every answer against the public contract:
// each is the Audit, its ETag and its preparation as of one commit, never
// a conflict (ASC-R149, V414-018).
func TestSharedPostgresAuditDetailReadsOneCommitWhileTheAuditChanges(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	trace := &auditListTrace{}
	tracedConfig := pool.Config()
	tracedConfig.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, tracedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	// The bearer principal of the contract requests owns the Audit.
	const owner, projectID, auditID = "user-1", "project-detail", "detail-preparing"
	f := newPreparedListFixture(t, ctx, pool, owner, projectID)
	f.start(projectID, auditID)
	f.claim(1)
	credentials := newFakeManagedCredentials()
	service, err := auditservice.New(auditservice.Options{Storage: artifactpgtest.NewStorage(), Pool: traced, Profiles: f.catalog, CredentialGuard: credentials,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return credentials, nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil,
		func(dependencies *testServices) { dependencies.Audits = service })
	router := publicContractRouter(t)
	type detail struct {
		Revision    uint64                    `json:"revision"`
		State       auditstore.AuditState     `json:"state"`
		Preparation *auditPreparationResponse `json:"preparation"`
	}
	// serve answers request while interleave commits on another connection
	// before the first statement that reads preparation, and checks the
	// answer against the contract and its ETag against its revision.
	serve := func(t *testing.T, request *http.Request, want int, interleave func()) detail {
		t.Helper()
		trace.mu.Lock()
		trace.marker, trace.interleaved, trace.err, trace.interleave = "LATERAL jsonb_each(", false, nil, nil
		if interleave != nil {
			trace.interleave = func(*pgx.Conn) error {
				interleave()
				return nil
			}
		}
		trace.mu.Unlock()
		response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
		if interleave != nil && (!trace.interleaved || trace.err != nil) {
			t.Fatalf("interleaving ran=%t err=%v", trace.interleaved, trace.err)
		}
		var read detail
		if response.Code != want || json.Unmarshal(response.Body.Bytes(), &read) != nil || read.Preparation == nil ||
			response.Header().Get("ETag") != strconv.Quote(strconv.FormatUint(read.Revision, 10)) {
			t.Fatalf("%s %s = %d ETag %s: %s", request.Method, request.URL.Path, response.Code,
				response.Header().Get("ETag"), response.Body.String())
		}
		return read
	}
	get := func() *http.Request { return jsonContractRequest(http.MethodGet, "/v1/audits/"+auditID, "") }
	role := func(read detail) auditPreparationRoleResponse { return read.Preparation.Roles["generate-api"] }

	// The first prepare attempt is created and submitted while the detail is
	// read: the answer is the Audit before it.
	pending := f.current(auditID)
	read := serve(t, get(), http.StatusOK, func() { f.attempt(projectID, auditID, "generate-api", 1, "submitted") })
	if read.Revision != pending.Revision || role(read).Status != auditdomain.PreparationPending || role(read).Attempts != 0 {
		t.Fatalf("detail across a submitted attempt = %+v, want revision %d pending", read, pending.Revision)
	}
	submitted := f.current(auditID)
	if submitted.Revision <= pending.Revision {
		t.Fatalf("the submitted attempt left revision %d", submitted.Revision)
	}
	read = serve(t, get(), http.StatusOK, nil)
	if read.Revision != submitted.Revision || role(read).Status != auditdomain.PreparationRunning || role(read).Attempts != 1 ||
		role(read).RunID == nil || len(role(read).Outputs) != 0 {
		t.Fatalf("detail after the submitted attempt = %+v, want revision %d running", read, submitted.Revision)
	}

	// The owner pauses the Audit and, before the response reads the detail,
	// the attempt succeeds and its outputs are accepted. The pause committed
	// and answers its own state, without the later acceptance.
	pause := jsonContractRequest(http.MethodPost, "/v1/audits/"+auditID+"/pause", "",
		"Idempotency-Key", "pause-detail", "If-Match", strconv.Quote(strconv.FormatUint(submitted.Revision, 10)))
	paused := serve(t, pause, http.StatusOK, func() { f.settle(projectID, auditID, "generate-api", 1, "succeeded") })
	accepted := f.current(auditID)
	if paused.State != auditstore.AuditPaused || paused.Revision <= submitted.Revision || paused.Revision >= accepted.Revision ||
		role(paused).Status != auditdomain.PreparationRunning || len(role(paused).Outputs) != 0 {
		t.Fatalf("pause across an accepted attempt = %+v; the Audit is at revision %d", paused, accepted.Revision)
	}
	read = serve(t, get(), http.StatusOK, nil)
	if read.Revision != accepted.Revision || read.State != auditstore.AuditPaused ||
		role(read).Status != auditdomain.PreparationAccepted || len(role(read).Outputs) != 2 {
		t.Fatalf("detail after the acceptance = %+v, want revision %d accepted", read, accepted.Revision)
	}
	// A replay of the pause answers the current Audit.
	replay := jsonContractRequest(http.MethodPost, "/v1/audits/"+auditID+"/pause", "",
		"Idempotency-Key", "pause-detail", "If-Match", strconv.Quote(strconv.FormatUint(submitted.Revision, 10)))
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, replay)
	if response.Code != http.StatusOK || response.Header().Get("Idempotency-Replayed") != "true" ||
		response.Header().Get("ETag") != strconv.Quote(strconv.FormatUint(accepted.Revision, 10)) {
		t.Fatalf("pause replay = %d ETag %s: %s", response.Code, response.Header().Get("ETag"), response.Body.String())
	}
}
