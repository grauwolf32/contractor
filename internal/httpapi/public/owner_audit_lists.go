package public

import (
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func parseAuditStates(value string) ([]auditstore.AuditState, error) {
	if value == "" {
		return nil, nil
	}
	parts := strings.Split(value, ",")
	if len(parts) > 10 {
		return nil, errInvalidRequest
	}
	states := make([]auditstore.AuditState, 0, len(parts))
	seen := make(map[auditstore.AuditState]bool)
	for _, part := range parts {
		state := auditstore.AuditState(part)
		if !state.Valid() {
			return nil, errInvalidRequest
		}
		if !seen[state] {
			states = append(states, state)
			seen[state] = true
		}
	}
	sort.Slice(states, func(i, j int) bool { return states[i] < states[j] })
	return states, nil
}

func auditStateKey(states []auditstore.AuditState) string {
	parts := make([]string, len(states))
	for i, state := range states {
		parts[i] = string(state)
	}
	return strings.Join(parts, ",")
}

func (h *handler) ownerListPosition(cursorValue, kind string) (*time.Time, string, error) {
	cursor, err := h.decodePageCursor(cursorValue, kind, 2)
	if err != nil || len(cursor) == 0 {
		return nil, "", err
	}
	createdAt, err := time.Parse(time.RFC3339Nano, cursor[0])
	if err != nil || createdAt.IsZero() {
		return nil, "", errInvalidRequest
	}
	return &createdAt, cursor[1], nil
}

func (h *handler) listOwnerFindings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	query, limit, cursor, err := pageQuery(r.URL.RawQuery, "state", "verdict", "severity", "auditState")
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditservice.OwnerFindingListParams{OwnerID: principalUserID(r.Context()), Limit: limit + 1}
	params.AuditStates, err = parseAuditStates(query.Get("auditState"))
	if err != nil {
		h.handleError(w, err)
		return
	}
	if values, ok := query["state"]; ok {
		state := auditservice.FindingState(values[0])
		if !state.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.State = &state
	}
	if values, ok := query["verdict"]; ok {
		if values[0] == "unreviewed" {
			params.Unreviewed = true
		} else {
			verdict := auditservice.AnalystVerdict(values[0])
			if verdict != auditservice.VerdictTruePositive && verdict != auditservice.VerdictFalsePositive {
				h.handleError(w, errInvalidRequest)
				return
			}
			params.Verdict = &verdict
		}
	}
	if values, ok := query["severity"]; ok {
		severity := auditservice.FindingSeverity(values[0])
		if !severity.Valid() || params.Unreviewed {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.Severity = &severity
	}
	kind := "owner-findings:" + params.OwnerID + ":" + query.Get("state") + ":" + query.Get("verdict") + ":" + query.Get("severity") + ":" + auditStateKey(params.AuditStates)
	params.BeforeCreatedAt, params.BeforeFindingID, err = h.ownerListPosition(cursor, kind)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items, err := h.dependencies.Audits.ListOwnerFindings(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items, page, err := paginate(h, items, limit, kind, func(last auditservice.Finding) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID}
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if items == nil {
		items = []auditservice.Finding{}
	}
	writeJSON(w, http.StatusOK, struct {
		Items []auditservice.Finding `json:"items"`
		Page  pageInfoResponse       `json:"page"`
	}{items, page})
}

func (h *handler) listOwnerReviews(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	query, limit, cursor, err := pageQuery(r.URL.RawQuery, "state", "auditState")
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditservice.OwnerReviewListParams{OwnerID: principalUserID(r.Context()), Limit: limit + 1}
	params.AuditStates, err = parseAuditStates(query.Get("auditState"))
	if err != nil {
		h.handleError(w, err)
		return
	}
	if values, ok := query["state"]; ok {
		state := auditservice.ReviewState(values[0])
		if !state.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.State = &state
	}
	kind := "owner-reviews:" + params.OwnerID + ":" + query.Get("state") + ":" + auditStateKey(params.AuditStates)
	params.BeforeCreatedAt, params.BeforeRequestID, err = h.ownerListPosition(cursor, kind)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items, err := h.dependencies.Audits.ListOwnerReviews(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items, page, err := paginate(h, items, limit, kind, func(last auditservice.ReviewRequest) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.RequestID}
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if items == nil {
		items = []auditservice.ReviewRequest{}
	}
	writeJSON(w, http.StatusOK, struct {
		Items []auditservice.ReviewRequest `json:"items"`
		Page  pageInfoResponse             `json:"page"`
	}{items, page})
}
