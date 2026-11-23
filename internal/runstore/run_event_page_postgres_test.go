package runstore

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresListRunEventsBoundsPageDataBytes pins API-R110 on the real
// store. The history consists of valid Planner plan events appended through
// AppendPlannerEvent, each sized exactly in the budget's measure, the octet
// length of the stored data text. A page returns the contiguous prefix within
// its event count and 8 MiB of data (LIM-API-24), reports a committed
// successor without transferring that event's data, measures at most one
// event beyond what it returns, and starts every fresh page with the next
// event however large a legal event is (LIM-API-25).
func TestPostgresListRunEventsBoundsPageDataBytes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	history := startSizedPlannerEvents(t, ctx, pool, store, "run-event-pages", "stage-event-pages")
	start := history.last
	const small = 4096
	largest := maxRunEventDataBytes
	p1 := history.append(small)
	p2 := history.append(largest)
	p3 := history.append(largest)
	p4 := history.append(largest)
	// From start, p1 to p5 exceed the budget by exactly one byte.
	p5 := history.append(largest - small + 1)
	// From p1, p2 to p6 fill the budget exactly.
	p6 := history.append(small - 1)
	p7 := history.append(small)
	if p1 != start+1 || p7 != start+7 {
		t.Fatalf("sized events got sequences %d..%d after %d", p1, p7, start)
	}
	sum := 0
	for _, sequence := range []int64{p1, p2, p3, p4, p5} {
		sum += history.size[sequence]
	}
	if sum != maxRunEventPageDataBytes+1 {
		t.Fatalf("p1..p5 hold %d data bytes, want one byte past the budget", sum)
	}

	sequences := func(first, last int64) []int64 {
		var result []int64
		for sequence := first; sequence <= last; sequence++ {
			result = append(result, sequence)
		}
		return result
	}
	for _, test := range []struct {
		name  string
		runID string
		after int64
		limit int
		want  []int64
		more  bool
		bytes int
	}{
		{"one byte past the budget", history.runID, start, 128, sequences(p1, p4), true, small + 3*largest},
		{"exactly the budget", history.runID, p1, 128, sequences(p2, p6), true, maxRunEventPageDataBytes},
		{"largest legal first event through the end", history.runID, p2, 128, sequences(p3, p7), false, 2*largest + largest + small},
		{"event count before the budget", history.runID, start, 2, sequences(p1, p2), true, small + largest},
		{"event count reached at the end", history.runID, p6, 1, sequences(p7, p7), false, small},
		{"empty after the end", history.runID, p7, 128, nil, false, 0},
		{"unknown Run", "run-event-pages-absent", 0, 128, nil, false, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			page, err := store.ListRunEvents(ctx, test.runID, test.after, test.limit)
			if err != nil {
				t.Fatal(err)
			}
			var got []int64
			dataBytes := 0
			for _, event := range page.Events {
				got = append(got, event.SequenceNumber)
				dataBytes += len(event.Data)
				if size, sized := history.size[event.SequenceNumber]; sized && len(event.Data) != size {
					t.Fatalf("event %d transferred %d data bytes, stored %d", event.SequenceNumber, len(event.Data), size)
				}
			}
			if !slices.Equal(got, test.want) || page.More != test.more || dataBytes != test.bytes ||
				dataBytes > maxRunEventPageDataBytes {
				t.Fatalf("page after %d = %v more=%t with %d data bytes, want %v more=%t with %d",
					test.after, got, page.More, dataBytes, test.want, test.more, test.bytes)
			}
			walked, lookAhead := walkRunEventPage(t, ctx, pool, test.runID, test.after, test.limit)
			wantWalked := len(test.want)
			if test.more {
				wantWalked++
			}
			if walked != wantWalked || walked > test.limit+1 || lookAhead != test.more {
				t.Fatalf("page statement walked %d events (look-ahead %t), want %d (look-ahead %t)",
					walked, lookAhead, wantWalked, test.more)
			}
		})
	}

	// Following More from the cursor 0 delivers the whole history once, in
	// order, without exceeding the budget on any page.
	var replayed []int64
	reads := 0
	for after, more := int64(0), true; more; reads++ {
		page, err := store.ListRunEvents(ctx, history.runID, after, 128)
		if err != nil || len(page.Events) == 0 && page.More {
			t.Fatalf("replay page after %d = (%d events, more=%t, %v)", after, len(page.Events), page.More, err)
		}
		dataBytes := 0
		for _, event := range page.Events {
			replayed = append(replayed, event.SequenceNumber)
			dataBytes += len(event.Data)
		}
		if dataBytes > maxRunEventPageDataBytes {
			t.Fatalf("replay page after %d returned %d data bytes", after, dataBytes)
		}
		if len(page.Events) != 0 {
			after = page.Events[len(page.Events)-1].SequenceNumber
		}
		more = page.More
	}
	// Pages: lifecycle and start events with p1..p4, p5..p7.
	if !slices.Equal(replayed, sequences(1, p7)) || reads != 2 {
		t.Fatalf("replay delivered %v in %d reads, want 1..%d in 2", replayed, reads, p7)
	}

	for _, limit := range []int{0, maxRunEventPageEvents + 1} {
		if _, err := store.ListRunEvents(ctx, history.runID, 0, limit); err == nil {
			t.Fatalf("ListRunEvents accepted limit %d", limit)
		}
	}
}

// TestPostgresListRunEventsFailsOnInvalidStoredEventsAndKeepsGaps pins the
// read validation of API-R110: a stored event that fails the append
// validation or does not fit an empty page fails the read that would return
// it instead of being skipped, a page never validates the data of its
// look-ahead, and a sequence gap reaches the caller, which treats it as a
// resync (API-R60).
func TestPostgresListRunEventsFailsOnInvalidStoredEventsAndKeepsGaps(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedRunStorePool(t, ctx)
	store := NewPostgresStore(pool, artifactpgtest.NewStorage())
	insert := func(runID string, sequence int64, kind RunEventKind, data string) {
		t.Helper()
		if _, err := pool.Exec(ctx, `
INSERT INTO workflow_run_events (run_id, sequence_number, event_id, event_schema_version, kind, data)
VALUES ($1, $2, $3, $4, $5, $6::jsonb)`,
			runID, sequence, fmt.Sprintf("%s-direct-%d", runID, sequence), contracts.APIVersion, kind, data,
		); err != nil {
			t.Fatalf("insert stored event %d of %s: %v", sequence, runID, err)
		}
	}
	lifecycle := func(runID string) string {
		return fmt.Sprintf(`{"runId":%q,"resource":"run","state":"running"}`, runID)
	}

	t.Run("sequence gap", func(t *testing.T) {
		const runID = "run-event-gap"
		createTestRun(t, ctx, store, runID)
		insert(runID, 3, RunEventLifecycleChanged, lifecycle(runID))
		page, err := store.ListRunEvents(ctx, runID, 0, 10)
		if err != nil || len(page.Events) != 2 || page.More ||
			page.Events[0].SequenceNumber != 1 || page.Events[1].SequenceNumber != 3 {
			t.Fatalf("page across a sequence gap = (%+v, %v)", page, err)
		}
	})

	t.Run("invalid stored event", func(t *testing.T) {
		const runID = "run-event-invalid"
		createTestRun(t, ctx, store, runID)
		insert(runID, 2, RunEventPlannerStarted, `{"unexpected":true}`)
		if page, err := store.ListRunEvents(ctx, runID, 0, 10); err == nil {
			t.Fatalf("page with an invalid stored event = %+v, want an error", page)
		}
		// As the look-ahead the event only shows that the page has a successor.
		page, err := store.ListRunEvents(ctx, runID, 0, 1)
		if err != nil || len(page.Events) != 1 || page.Events[0].SequenceNumber != 1 || !page.More {
			t.Fatalf("page before an invalid stored event = (%+v, %v)", page, err)
		}
		if page, err := store.ListRunEvents(ctx, runID, 1, 10); err == nil {
			t.Fatalf("page starting at an invalid stored event = %+v, want an error", page)
		}
	})

	// Only a store without the per-event CHECK can hold these events, plan
	// events whose only defect is their size.
	if _, err := pool.Exec(ctx, `ALTER TABLE workflow_run_events DROP CONSTRAINT workflow_run_events_data_check`); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		dataBytes int
	}{
		{"event above the per-event bound", maxRunEventDataBytes + 1},
		{"event above the page budget", maxRunEventPageDataBytes + 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			runID := "run-event-" + strings.ReplaceAll(test.name, " ", "-")
			createTestRun(t, ctx, store, runID)
			identity := plannerRunEventData{
				StageExecutionID: "stage-" + runID, SessionID: "session-" + runID, InvocationID: "invocation-" + runID,
			}
			data := sizedPlanEventData(t, ctx, pool, identity, test.dataBytes)
			insert(runID, 2, RunEventPlannerPlanChanged, string(data))
			if page, err := store.ListRunEvents(ctx, runID, 1, 10); err == nil {
				t.Fatalf("page starting at an oversized stored event = %d events more=%t, want an error",
					len(page.Events), page.More)
			}
			// As the look-ahead the event only shows that the page has a successor.
			page, err := store.ListRunEvents(ctx, runID, 0, 1)
			if err != nil || len(page.Events) != 1 || !page.More {
				t.Fatalf("page before an oversized stored event = (%d events, more=%t, %v)",
					len(page.Events), page.More, err)
			}
		})
	}
}

// sizedPlannerEvents appends valid Planner plan-changed events of an exact
// stored data size to one Run through the production append path.
type sizedPlannerEvents struct {
	t        *testing.T
	ctx      context.Context
	pool     *pgxpool.Pool
	store    *PostgresStore
	runID    string
	identity plannerRunEventData
	// nextSessionEvent is the Planner session sequence of the next event.
	nextSessionEvent int64
	// last is the Run event sequence of the latest event.
	last int64
	// size maps the Run event sequence of each sized event to its stored
	// data bytes.
	size map[int64]int
}

func startSizedPlannerEvents(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, store *PostgresStore, runID, stageID string,
) *sizedPlannerEvents {
	t.Helper()
	createPreparingStage(t, ctx, store, runID, stageID)
	params := testStartPlannerParams(t, stageID, "session-"+runID)
	if err := store.StartPlanner(ctx, params); err != nil {
		t.Fatal(err)
	}
	history := &sizedPlannerEvents{
		t: t, ctx: ctx, pool: pool, store: store, runID: runID,
		identity: plannerRunEventData{
			StageExecutionID: params.StageExecutionID, SessionID: params.SessionID, InvocationID: params.InvocationID,
		},
		nextSessionEvent: 2, size: make(map[int64]int),
	}
	history.last = history.cursor()
	return history
}

// append stores the next plan-changed event with dataBytes of stored data and
// returns its Run event sequence.
func (h *sizedPlannerEvents) append(dataBytes int) int64 {
	h.t.Helper()
	data := sizedPlanEventData(h.t, h.ctx, h.pool, h.identity, dataBytes)
	err := h.store.AppendPlannerEvent(h.ctx, AppendPlannerEventParams{
		SessionID: h.identity.SessionID, StageExecutionID: h.identity.StageExecutionID,
		InvocationID: h.identity.InvocationID, SequenceNumber: h.nextSessionEvent,
		NewStateSchemaVersion: contracts.APIVersion,
		NewState:              json.RawMessage(fmt.Sprintf(`{"step":%d}`, h.nextSessionEvent)),
		RunEvent: RunEventAppend{
			EventID:            fmt.Sprintf("%s-sized-%d", h.runID, h.nextSessionEvent),
			EventSchemaVersion: contracts.APIVersion, Kind: RunEventPlannerPlanChanged, Data: data,
		},
	})
	if err != nil {
		h.t.Fatalf("append a Planner event of %d data bytes: %v", dataBytes, err)
	}
	h.nextSessionEvent++
	sequence := h.cursor()
	if sequence != h.last+1 {
		h.t.Fatalf("appended event got sequence %d after %d", sequence, h.last)
	}
	h.last = sequence
	h.size[sequence] = dataBytes
	return sequence
}

// sizedPlanEventData builds plan-changed data for identity whose stored text
// is exactly dataBytes long and which is otherwise valid. Padding the
// instructions with ASCII adds one byte per character to the stored text, so
// the stored size of an unpadded plan fixes the padding.
func sizedPlanEventData(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, identity plannerRunEventData, dataBytes int,
) json.RawMessage {
	t.Helper()
	const room = maxRunEventSubtaskInstructionBytes - 1
	subtasks := dataBytes/room + 1
	build := func(padding int) json.RawMessage {
		plan := plannerRunPlanProjection{Revision: 1, CurrentSubtaskID: "0"}
		for index := range subtasks {
			extra := min(padding, room)
			padding -= extra
			plan.Subtasks = append(plan.Subtasks, plannerRunSubtask{
				ID: strconv.Itoa(index), Objective: "sized objective",
				Instructions: strings.Repeat("x", 1+extra), Status: "pending",
			})
		}
		data := identity
		data.Plan = &plan
		return mustRunEventJSON(t, data)
	}
	stored := func(data json.RawMessage) int {
		var result int
		if err := pool.QueryRow(ctx, `SELECT octet_length($1::jsonb::text)`, string(data)).Scan(&result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	padding := dataBytes - stored(build(0))
	if padding < 0 || padding > subtasks*room {
		t.Fatalf("cannot size a plan event to %d data bytes", dataBytes)
	}
	data := build(padding)
	if got := stored(data); got != dataBytes {
		t.Fatalf("sized plan event stores %d data bytes, want %d", got, dataBytes)
	}
	return data
}

func (h *sizedPlannerEvents) cursor() int64 {
	h.t.Helper()
	cursor, err := h.store.GetRunEventCursor(h.ctx, h.runID)
	if err != nil {
		h.t.Fatal(err)
	}
	return cursor.Sequence
}

// walkRunEventPage runs the page statement twice: under EXPLAIN ANALYZE, for
// the number of events its walk measured, the rows of its recursive union,
// and as is, for whether its rows end with a look-ahead and whether the
// look-ahead carries no data.
func walkRunEventPage(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string, after int64, limit int,
) (walked int, lookAhead bool) {
	t.Helper()
	var encoded []byte
	if err := pool.QueryRow(ctx, "EXPLAIN (ANALYZE, FORMAT JSON) "+listRunEventsSQL,
		runID, after, limit, maxRunEventPageDataBytes).Scan(&encoded); err != nil {
		t.Fatalf("explain the page statement: %v", err)
	}
	type node struct {
		NodeType   string  `json:"Node Type"`
		ActualRows float64 `json:"Actual Rows"`
		Plans      []node  `json:"Plans"`
	}
	var plans []struct {
		Plan node `json:"Plan"`
	}
	if err := json.Unmarshal(encoded, &plans); err != nil || len(plans) != 1 {
		t.Fatalf("decode the page plan: %v", err)
	}
	unions := 0
	var visit func(node)
	visit = func(current node) {
		if current.NodeType == "Recursive Union" {
			unions++
			walked = int(current.ActualRows)
		}
		for _, child := range current.Plans {
			visit(child)
		}
	}
	visit(plans[0].Plan)
	if unions != 1 {
		t.Fatalf("page plan has %d recursive unions, want 1", unions)
	}

	rows, err := pool.Query(ctx, listRunEventsSQL, runID, after, limit, maxRunEventPageDataBytes)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	statementRows := 0
	for rows.Next() {
		values, err := rows.Values()
		if err != nil {
			t.Fatal(err)
		}
		statementRows++
		returned, _ := values[6].(bool)
		if lookAhead || !returned && values[7] != nil || returned && values[7] == nil {
			t.Fatalf("page statement row %d returned=%t with data %t after look-ahead %t",
				statementRows, returned, values[7] != nil, lookAhead)
		}
		lookAhead = !returned
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if statementRows != walked {
		t.Fatalf("page statement returned %d rows for %d walked events", statementRows, walked)
	}
	return walked, lookAhead
}
