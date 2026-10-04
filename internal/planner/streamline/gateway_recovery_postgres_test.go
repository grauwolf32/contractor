package streamline

import (
	"context"
	cryptorand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestPostgresGatewayRecoveryEndsPermanentFailuresNonRetryable(t *testing.T) {
	const secret = "sk-provider-echoed-secret"
	contextLength := `{"error":{"code":"context_length_exceeded","message":"prompt ` + secret + ` is too long"}}`
	for _, test := range []struct {
		name      string
		responses []int
		body      string
		wantCode  string
	}{
		{name: "access denied", responses: []int{http.StatusUnauthorized}, wantCode: "gateway_access_denied"},
		{name: "context length", responses: []int{http.StatusBadRequest}, body: contextLength, wantCode: "context_length_exceeded"},
		{name: "retried outage then access denied", responses: []int{http.StatusServiceUnavailable, http.StatusUnauthorized},
			wantCode: "gateway_access_denied"},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedStreamlinePool(t, ctx)
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				call := int(calls.Add(1))
				if call > len(test.responses) {
					t.Errorf("unexpected Gateway call %d", call)
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.responses[call-1])
				body := test.body
				if body == "" {
					body = `{"error":{"message":"credential ` + secret + ` rejected"}}`
				}
				_, _ = w.Write([]byte(body))
			}))
			t.Cleanup(server.Close)
			participant := admittedPlannerParticipant(t, ctx, pool, "run-gateway")
			sessions := newFakeSessions()
			instance := recoveryBackedPlanner(t, server, participant, sessions, 32)

			_, err := instance.Run(ctx)
			assertPlannerCode(t, err, "planner_gateway_rejected")
			failure := planner.FailureFrom(err)
			if failure.Retryable || !strings.Contains(failure.Message, test.wantCode) ||
				int(calls.Load()) != len(test.responses) {
				t.Fatalf("failure=%+v Gateway calls=%d", failure, calls.Load())
			}
			if sessions.completion == nil || sessions.completion.Failure == nil ||
				*sessions.completion.Failure != failure {
				t.Fatalf("recorded completion = %+v, want failure %+v", sessions.completion, failure)
			}
			report, _ := instance.(planner.ReportProvider).ExecutionReport()
			encoded, marshalErr := json.Marshal(struct {
				Error  string                    `json:"error"`
				Report contracts.ExecutionReport `json:"report"`
			}{Error: err.Error(), Report: report})
			if marshalErr != nil || strings.Contains(string(encoded), secret) {
				t.Fatalf("provider content leaked: %s (%v)", encoded, marshalErr)
			}
			// The permanent answer of the recovery probe proves the transport
			// recovered: the route reopens and the Run stops waiting.
			var blocked bool
			var state string
			if err := pool.QueryRow(ctx, `
SELECT g.blocked, r.state FROM gateway_recovery_routes g, workflow_runs r
WHERE r.run_id='run-gateway'`).Scan(&blocked, &state); err != nil || blocked || state != string(runstore.RunRunning) {
				t.Fatalf("recovery state blocked=%v Run=%s err=%v", blocked, state, err)
			}
		})
	}
}

// admittedPlannerParticipant binds a planner invocation of one admitted,
// running Run to the production recovery authority.
func admittedPlannerParticipant(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string,
) *gatewayrecovery.Participant {
	t.Helper()
	service, err := gatewayrecovery.New(pool, gatewayrecovery.Policy{
		RequestTimeout: time.Minute, InitialDelay: time.Second, MaxDelay: time.Second, AutomaticWindow: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	if _, err := runs.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "owner", WorkflowName: "test", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"}); err != nil {
		t.Fatal(err)
	}
	route := gatewayrecovery.Route{OwnerID: "owner", GatewayDigest: "gateway", Model: "planner-model"}
	if allowed, err := service.Admit(ctx, runID, []gatewayrecovery.Route{route}); err != nil || !allowed {
		t.Fatalf("admit route = %v, %v", allowed, err)
	}
	if _, err := runs.TransitionRun(ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"}); err != nil {
		t.Fatal(err)
	}
	return service.Planner(runID, "stage-1", route, contracts.DefaultGatewayFailureSignatures())
}

func recoveryBackedPlanner(
	t *testing.T, server *httptest.Server, participant *gatewayrecovery.Participant,
	sessions *fakeSessions, maxModelCalls int,
) planner.Planner {
	t.Helper()
	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		Recovery: participant, URL: server.URL + "/v1", Model: "planner-model", HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatal(err)
	}
	invocation := testInvocation("builder")
	invocation.ModelAccess.ModelPolicy.MaxModelCalls = maxModelCalls
	instance, err := mustFactory(t, sessions, &fakeWorkerInvoker{}, &fakeInspector{}, llm, Limits{}).Create(invocation)
	if err != nil {
		t.Fatal(err)
	}
	return instance
}

func isolatedStreamlinePool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	random := make([]byte, 8)
	if _, err := cryptorand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	schema := "contractor_streamline_" + hex.EncodeToString(random)
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanup, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop Streamline test schema: %v", err)
		}
		admin.Close()
	})
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatal(err)
	}
	return pool
}
