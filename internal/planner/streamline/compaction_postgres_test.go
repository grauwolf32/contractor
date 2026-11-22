package streamline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/planner"
	plannersession "github.com/grauwolf32/contractor/internal/planner/session"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/stageresult"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
)

// markerPredicate selects the content-free checkpoint marker rows: Planner
// activity without function calls, results or model usage.
func markerPredicate(prefix string) string {
	return fmt.Sprintf(`%[1]skind = 'planner.activity'
 AND %[1]sdata #>> '{activity,author}' IN ('streamline_planner', 'router_planner')
 AND %[1]sdata #> '{activity,functionCalls}' = 'null'::jsonb
 AND %[1]sdata #> '{activity,functionResults}' = 'null'::jsonb
 AND NOT (%[1]sdata -> 'activity' ? 'inputTokens')`, prefix)
}

// markerProbeSessions forwards to the real PostgreSQL-backed session and
// counts checkpoint marker appends; lostAck reports a committed marker
// append as failed.
type markerProbeSessions struct {
	*plannersession.Service
	mu      sync.Mutex
	markers int
	lostAck bool
}

func (s *markerProbeSessions) NewADKSession(
	ctx context.Context, identity planner.SessionIdentity, options plannersession.ADKOptions,
) (adksession.Service, error) {
	service, err := s.Service.NewADKSession(ctx, identity, options)
	if err != nil {
		return nil, err
	}
	return &markerProbeADK{Service: service, owner: s}, nil
}

type markerProbeADK struct {
	adksession.Service
	owner *markerProbeSessions
}

func (s *markerProbeADK) AppendEvent(ctx context.Context, current adksession.Session, event *adksession.Event) error {
	marker := event != nil && event.Content == nil && event.Author != "user"
	err := s.Service.AppendEvent(ctx, current, event)
	if marker {
		s.owner.mu.Lock()
		s.owner.markers++
		lost := s.owner.lostAck
		s.owner.mu.Unlock()
		if err == nil && lost {
			return io.ErrUnexpectedEOF
		}
	}
	return err
}

func (s *markerProbeSessions) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.markers
}

func postgresCompactionFactory(
	t *testing.T, profile string, sessions *markerProbeSessions, workers planner.WorkerInvoker,
	main *scriptedModel, compactor *scriptedCompactor,
) *Factory {
	t.Helper()
	create := NewConfiguredFactory
	if profile == "router" {
		create = NewConfiguredRouterDelegate
	}
	factory, err := create(
		sessions, sessions, workers, compactionInspector(), unavailableWorkerStateReader{},
		compactionModelFactory(main, compactor), Limits{MaxWallTime: time.Minute},
	)
	if err != nil {
		t.Fatal(err)
	}
	return factory
}

func compactionPostgresStore(t *testing.T, ctx context.Context) (*pgxpool.Pool, *runstore.PostgresStore, *markerProbeSessions) {
	t.Helper()
	pool := isolatedStreamlinePool(t, ctx)
	store := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	createPreparingStreamlineStage(t, ctx, store)
	service, err := plannersession.New(store, plannersession.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return pool, store, &markerProbeSessions{Service: service}
}

func TestPostgresCompactionTwoCheckpointsKeepRawContextOutOfDurableFacts(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool, store, sessions := compactionPostgresStore(t, ctx)
			log := &requestLog{}
			llm := &scriptedModel{steps: twoCompactionSteps(profile, log)}
			compactor := &scriptedCompactor{steps: []compactorStep{summaryStep("summary-one"), summaryStep("summary-two")}}
			workers := &sequenceWorkers{}
			factory := postgresCompactionFactory(t, profile, sessions, workers, llm, compactor)
			invocation := compactionInvocation(profile)
			instance, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(ctx)
			if err != nil {
				t.Fatal(err)
			}
			assertCompactionReleased(t, instance)
			if result.Outcome != stageresult.StageSucceeded || refKey(result.Artifacts["report"]) != "builder/review@report-r2" {
				t.Fatalf("result = %+v", result)
			}
			if llm.callCount() != 6 || workers.count() != 2 || len(compactor.calls()) != 2 || sessions.count() != 2 {
				t.Fatalf("main/workers/compactor/markers = %d/%d/%d/%d", llm.callCount(), workers.count(), len(compactor.calls()), sessions.count())
			}
			var markers int
			var raw string
			if err := pool.QueryRow(ctx, `SELECT count(*) FILTER (WHERE `+markerPredicate("")+`), string_agg(data::text, '')
FROM workflow_run_events`).Scan(&markers, &raw); err != nil {
				t.Fatal(err)
			}
			if markers != 2 {
				t.Fatalf("durable checkpoint markers = %d, want 2", markers)
			}
			for _, private := range []string{"summary-one", "summary-two", "Context checkpoint", "validated result", "Keep decisions"} {
				if strings.Contains(raw, private) {
					t.Fatalf("raw context %q reached durable facts", private)
				}
			}
			stored := loadAppendProbeCompletion(t, ctx, store)
			if stored.Result == nil || stored.Failure != nil {
				t.Fatal("completed candidate was not stored")
			}
			// Completed recovery is model-free: no model, compactor or Worker call.
			recovered, err := factory.Create(invocation)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := recovered.Run(ctx); err != nil {
				t.Fatal(err)
			}
			if llm.callCount() != 6 || workers.count() != 2 || len(compactor.calls()) != 2 {
				t.Fatal("completed recovery repeated a model, compactor or Worker call")
			}
		})
	}
}

// A marker append the store refuses, an outage or a lost acknowledgement
// fences continuation: no further model or Worker call and no candidate.
func TestPostgresCompactionMarkerWriteFailureFences(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		for _, fault := range []string{"invalid", "unavailable", "lost-ack"} {
			t.Run(profile+"/"+fault, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
				defer cancel()
				pool, store, sessions := compactionPostgresStore(t, ctx)
				switch fault {
				case "invalid":
					if _, err := pool.Exec(ctx, `ALTER TABLE workflow_run_events ADD CONSTRAINT v420_marker_rejected CHECK (NOT (`+markerPredicate("")+`))`); err != nil {
						t.Fatal(err)
					}
				case "unavailable":
					statement := fmt.Sprintf(`CREATE FUNCTION v420_marker_outage() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF %s THEN
        RAISE EXCEPTION 'controlled marker outage' USING ERRCODE = '%s';
    END IF;
    RETURN NEW;
END $$`, markerPredicate("NEW."), persistencepostgres.SQLStateConnectionFailure)
					if _, err := pool.Exec(ctx, statement); err != nil {
						t.Fatal(err)
					}
					if _, err := pool.Exec(ctx, `CREATE TRIGGER v420_marker_outage BEFORE INSERT ON workflow_run_events FOR EACH ROW EXECUTE FUNCTION v420_marker_outage()`); err != nil {
						t.Fatal(err)
					}
				case "lost-ack":
					sessions.lostAck = true
				}
				log := &requestLog{}
				llm := &scriptedModel{steps: twoCompactionSteps(profile, log)}
				compactor := &scriptedCompactor{steps: []compactorStep{summaryStep("summary-one")}}
				workers := &sequenceWorkers{}
				instance, err := postgresCompactionFactory(t, profile, sessions, workers, llm, compactor).Create(compactionInvocation(profile))
				if err != nil {
					t.Fatal(err)
				}
				result, runErr := instance.Run(ctx)
				assertCompactionReleased(t, instance)
				wantCode, wantRetryable := "planner_session_unavailable", true
				if fault == "invalid" {
					wantCode, wantRetryable = "planner_session_rejected", false
					if !errors.Is(runErr, runstore.ErrInvalid) {
						t.Error("rejected marker lost errors.Is(runstore.ErrInvalid)")
					}
				}
				assertAppendProbeFailure(t, result, runErr, wantCode, wantRetryable)
				if llm.callCount() != 3 || workers.count() != 1 || len(compactor.calls()) != 1 || sessions.count() != 1 {
					t.Fatalf("main/workers/compactor/markers = %d/%d/%d/%d", llm.callCount(), workers.count(), len(compactor.calls()), sessions.count())
				}
				stored := loadAppendProbeCompletion(t, ctx, store)
				if stored.Failure == nil || stored.Failure.Code != wantCode || stored.Result != nil {
					t.Fatalf("stored completion = %s", appendProbeCompletionCode(stored))
				}
				var markers int
				if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_run_events WHERE `+markerPredicate("")).Scan(&markers); err != nil {
					t.Fatal(err)
				}
				// A lost acknowledgement keeps its one real commit; nothing replays it.
				wantMarkers := 0
				if fault == "lost-ack" {
					wantMarkers = 1
				}
				if markers != wantMarkers {
					t.Fatalf("durable markers = %d, want %d", markers, wantMarkers)
				}
			})
		}
	}
}

// A finish whose required append fails after two checkpoints still cannot
// become a completed candidate (PLN-R130).
func TestPostgresCompactionFinishAppendFailureAfterCheckpoints(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			pool, store, sessions := compactionPostgresStore(t, ctx)
			installRequiredEventGuard(t, ctx, pool, "finish", "invalid")
			log := &requestLog{}
			llm := &scriptedModel{steps: twoCompactionSteps(profile, log)}
			compactor := &scriptedCompactor{steps: []compactorStep{summaryStep("summary-one"), summaryStep("summary-two")}}
			workers := &sequenceWorkers{}
			instance, err := postgresCompactionFactory(t, profile, sessions, workers, llm, compactor).Create(compactionInvocation(profile))
			if err != nil {
				t.Fatal(err)
			}
			result, runErr := instance.Run(ctx)
			assertCompactionReleased(t, instance)
			assertAppendProbeFailure(t, result, runErr, "planner_session_rejected", false)
			if llm.callCount() != 6 || workers.count() != 2 || len(compactor.calls()) != 2 || sessions.count() != 2 {
				t.Fatalf("main/workers/compactor/markers = %d/%d/%d/%d", llm.callCount(), workers.count(), len(compactor.calls()), sessions.count())
			}
			stored := loadAppendProbeCompletion(t, ctx, store)
			if stored.Result != nil || stored.Failure == nil || stored.Failure.Code != "planner_session_rejected" {
				t.Fatalf("stored completion = %s", appendProbeCompletionCode(stored))
			}
		})
	}
}

func TestPostgresRollingCompactionAcceptsSmallerCombinedContext(t *testing.T) {
	for _, profile := range []string{"streamline", "router"} {
		t.Run(profile, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			defer cancel()
			_, store, sessions := compactionPostgresStore(t, ctx)
			step := func(rolling bool) compactorStep {
				return func(request *model.LLMRequest) (compactorReply, error) {
					var source compactorInput
					for _, content := range request.Contents {
						for _, part := range content.Parts {
							if part.Text != "" && json.Unmarshal([]byte(part.Text), &source) == nil && len(source.History) > 0 {
								var rawBytes int
								for _, event := range source.History {
									encoded, err := json.Marshal(event)
									if err != nil {
										t.Fatal(err)
									}
									rawBytes += len(encoded)
								}
								size := rawBytes / 2
								if rolling {
									size = rawBytes + 1
									if len(source.PreviousSummary) <= 1 {
										t.Fatal("second checkpoint has no useful prior summary")
									}
								}
								return summaryStep(strings.Repeat("s", size))(request)
							}
						}
					}
					t.Fatal("compactor did not receive its history")
					return compactorReply{}, nil
				}
			}
			compactor := &scriptedCompactor{steps: []compactorStep{step(false), step(true)}}
			llm := &scriptedModel{steps: twoCompactionSteps(profile, &requestLog{})}
			workers := &sequenceWorkers{}
			instance, err := postgresCompactionFactory(t, profile, sessions, workers, llm, compactor).Create(compactionInvocation(profile))
			if err != nil {
				t.Fatal(err)
			}
			result, err := instance.Run(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if result.Outcome != stageresult.StageSucceeded || sessions.count() != 2 || workers.count() != 2 {
				t.Fatalf("rolling continuation did not finish: result=%+v markers=%d workers=%d", result, sessions.count(), workers.count())
			}
			if completion := loadAppendProbeCompletion(t, ctx, store); completion.Result == nil || completion.Failure != nil {
				t.Fatal("successful candidate was not persisted")
			}
		})
	}
}
