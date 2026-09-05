package public

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (h *handler) getOwnerQueueControl(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	control, err := h.dependencies.Runs.GetOwnerQueueControl(
		r.Context(), principalUserID(r.Context()),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeOwnerQueueControl(w, http.StatusOK, control)
}

func (h *handler) putOwnerQueueControl(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, fmt.Errorf("%w: Queue control requires application/json", errInvalidRequest))
		return
	}
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		h.handleError(w, fmt.Errorf("%w: Queue control requires one If-Match", errInvalidRequest))
		return
	}
	revision, err := parseOwnerQueueControlETag(r.Header.Values("If-Match")[0])
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request updateOwnerQueueControlRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	if request.Paused == nil {
		h.handleError(w, fmt.Errorf("%w: Queue control paused is required", errInvalidRequest))
		return
	}
	control, err := h.dependencies.Runs.UpdateOwnerQueueControl(
		r.Context(),
		runstore.UpdateOwnerQueueControlParams{
			OwnerID: principalUserID(r.Context()), ExpectedRevision: revision,
			Paused: *request.Paused,
		},
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.dependencies.RunNotifier.Wake()
	writeOwnerQueueControl(w, http.StatusOK, control)
}

func parseOwnerQueueControlETag(raw string) (uint64, error) {
	if strings.HasPrefix(strings.TrimSpace(raw), "W/") {
		return 0, fmt.Errorf("%w: Queue control requires one strong ETag", errInvalidRequest)
	}
	value := strings.TrimSpace(raw)
	revision, err := strconv.Unquote(value)
	if err != nil || revision == "" {
		return 0, fmt.Errorf("%w: Queue control revision must be quoted", errInvalidRequest)
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil || strconv.FormatUint(parsed, 10) != revision {
		return 0, fmt.Errorf("%w: Queue control revision is invalid", errInvalidRequest)
	}
	return parsed, nil
}

func writeOwnerQueueControl(w http.ResponseWriter, status int, control runstore.OwnerQueueControl) {
	revision := strconv.FormatUint(control.Revision, 10)
	w.Header().Set("ETag", strconv.Quote(revision))
	response := ownerQueueControlResponse{Paused: control.Paused, Revision: revision}
	if !control.UpdatedAt.IsZero() {
		updatedAt := control.UpdatedAt
		response.UpdatedAt = &updatedAt
	}
	writeJSON(w, status, response)
}

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
