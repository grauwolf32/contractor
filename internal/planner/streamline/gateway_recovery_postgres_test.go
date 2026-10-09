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
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
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
			instance := recoveryBackedPlanner(t, server, participant, sessions, 32, 0)

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
				Report reporting.ExecutionReport `json:"report"`
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

func TestPostgresGatewayRecoveryKeepsOutcomeWhenBookkeepingWaits(t *testing.T) {
	for _, test := range []struct {
		name     string
		status   int
		wantCode string
		wantUsed int64
	}{
		// One allowed model call: the delivered reply is charged, then the
		// call budget ends the invocation.
		{name: "reply", status: http.StatusOK, wantCode: "planner_model_call_limit", wantUsed: 70},
		{name: "permanent rejection", status: http.StatusUnauthorized, wantCode: "planner_gateway_rejected"},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool := isolatedStreamlinePool(t, ctx)
			config := pool.Config()
			config.ConnConfig.RuntimeParams["lock_timeout"] = "200"
			budgeted, err := pgxpool.NewWithConfig(ctx, config)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(budgeted.Close)
			holders := make(chan pgx.Tx, 1)
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				// The request was admitted. Hold the Run row lock that recording
				// its outcome needs until the planner has returned.
				holder, err := pool.Begin(ctx)
				if err == nil {
					_, err = holder.Exec(ctx, `SELECT 1 FROM workflow_runs WHERE run_id='run-bookkeeping' FOR UPDATE`)
					holders <- holder
				}
				if err != nil {
					t.Errorf("hold Run lock: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(`{
  "model":"planner-model",
  "choices":[{"finish_reason":"stop","message":{"content":"still planning"}}],
  "usage":{"prompt_tokens":30,"completion_tokens":40,"total_tokens":70}
}`))
			}))
			t.Cleanup(server.Close)
			participant := admittedPlannerParticipant(t, ctx, budgeted, "run-bookkeeping")
			instance := recoveryBackedPlanner(t, server, participant, newFakeSessions(), 1, 0)

			_, runErr := instance.Run(ctx)
			select {
			case holder := <-holders:
				if err := holder.Rollback(ctx); err != nil {
					t.Fatal(err)
				}
			default:
				t.Fatal("Gateway request did not hold the Run lock")
			}
			assertPlannerCode(t, runErr, test.wantCode)
			report, _ := instance.(planner.ReportProvider).ExecutionReport()
			if report.Metrics.ModelCalls == nil || *report.Metrics.ModelCalls != 1 ||
				report.Metrics.TotalTokens == nil || *report.Metrics.TotalTokens != test.wantUsed || calls.Load() != 1 {
				t.Fatalf("report metrics=%+v Gateway calls=%d", report.Metrics, calls.Load())
			}
		})
	}
}

func TestPostgresGatewayRecoveryDoesNotResendAbandonedRequest(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedStreamlinePool(t, ctx)
	release := make(chan struct{})
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
		// Received but unanswered: a local model still generating.
		<-release
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(release) })
	participant := admittedPlannerParticipant(t, ctx, pool, "run-abandoned")
	instance := recoveryBackedPlanner(t, server, participant, newFakeSessions(), 32, 300*time.Millisecond)

	_, err := instance.Run(ctx)
	assertPlannerCode(t, err, "planner_gateway_unavailable")
	if failure := planner.FailureFrom(err); !failure.Retryable || calls.Load() != 1 {
		t.Fatalf("failure=%+v Gateway calls=%d, want one retryable unanswered request", failure, calls.Load())
	}
	// A slow model is not an outage: the route stays open and the Run running.
	var blocked bool
	var state string
	if err := pool.QueryRow(ctx, `
SELECT g.blocked, r.state FROM gateway_recovery_routes g, workflow_runs r
WHERE r.run_id='run-abandoned'`).Scan(&blocked, &state); err != nil || blocked || state != string(runstore.RunRunning) {
		t.Fatalf("recovery state blocked=%v Run=%s err=%v", blocked, state, err)
	}
}

// admittedPlannerParticipant binds a planner invocation of one admitted,
// running Run to the production recovery authority.
func admittedPlannerParticipant(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string,
) *gatewayrecovery.Participant {
	t.Helper()
	service, err := gatewayrecovery.New(pool, gatewayrecovery.Policy{
		InitialDelay: time.Second, MaxDelay: time.Second, AutomaticWindow: time.Minute,
	}, time.Minute)
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
	return service.Planner(runID, "stage-1", route, llmgateway.DefaultGatewayFailureSignatures())
}

func recoveryBackedPlanner(
	t *testing.T, server *httptest.Server, participant *gatewayrecovery.Participant,
	sessions *fakeSessions, maxModelCalls int, requestTimeout time.Duration,
) planner.Planner {
	t.Helper()
	llm, err := NewOpenAICompatibleModel(GatewaySettings{
		Recovery: participant, URL: server.URL + "/v1", Model: "planner-model", HTTPClient: server.Client(),
		RequestTimeout: requestTimeout,
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
