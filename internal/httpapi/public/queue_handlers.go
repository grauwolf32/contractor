package public

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (h *handler) listRunQueue(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "state", "membership")
	if err != nil {
		h.handleError(w, err)
		return
	}
	var state *runstore.WorkflowRunState
	if values, present := query["state"]; present {
		candidate := runstore.WorkflowRunState(values[0])
		if candidate != runstore.RunInitializing &&
			candidate != runstore.RunRunning &&
			candidate != runstore.RunCancelling {
			h.handleError(w, fmt.Errorf("%w: Queue state must be non-terminal", errInvalidRequest))
			return
		}
		state = &candidate
	}
	var membership *runstore.RunQueueMembership
	if values, present := query["membership"]; present {
		candidate := runstore.RunQueueMembership(values[0])
		if !candidate.Valid() {
			h.handleError(w, fmt.Errorf("%w: Queue membership is invalid", errInvalidRequest))
			return
		}
		membership = &candidate
	}
	cursorKind := runQueueCursorKind(state, membership)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := runstore.ListRunQueueParams{
		OwnerID: principalUserID(r.Context()), State: state,
		Membership: membership, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		after, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, fmt.Errorf("%w: invalid Queue cursor", errInvalidRequest))
			return
		}
		params.AfterCreatedAt = &after
		params.AfterRunID = cursor[1]
	}
	runs, err := h.dependencies.Runs.ListRunQueue(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(runs) > limit {
		runs = runs[:limit]
		last := runs[len(runs)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.RunID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	items := make([]queueItemResponse, 0, len(runs))
	for _, run := range runs {
		item := queueItemResponse{
			RunID: run.RunID, Workflow: run.WorkflowName + "@" + run.WorkflowVersion,
			State: run.State, Labels: run.MetadataLabels.Clone(),
			EventCursor: eventCursorResponse{
				Generation: run.EventCursor.Generation,
				Sequence:   strconv.FormatInt(run.EventCursor.Sequence, 10),
			},
			CreatedAt: run.CreatedAt, UpdatedAt: run.UpdatedAt,
		}
		if run.ProjectID != nil {
			item.Project = &queueProjectResponse{
				ProjectID: *run.ProjectID, Name: run.ProjectName,
				Kind: projectstore.Kind(run.ProjectKind),
			}
		}
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, queuePageResponse{Items: items, Page: page})
}

func runQueueCursorKind(
	state *runstore.WorkflowRunState,
	membership *runstore.RunQueueMembership,
) string {
	kind := "run-queue"
	if state != nil {
		kind += ":state:" + string(*state)
	}
	if membership != nil {
		kind += ":membership:" + string(*membership)
	}
	return kind
}
