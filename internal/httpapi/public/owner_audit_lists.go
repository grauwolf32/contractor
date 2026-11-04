package public

import (
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
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

func (h *auditHandlers) ownerListPosition(cursorValue, kind string) (*time.Time, string, error) {
	cursor, err := h.cursors.decodePageCursor(cursorValue, kind, 2)
	if err != nil || len(cursor) == 0 {
		return nil, "", err
	}
	createdAt, err := time.Parse(time.RFC3339Nano, cursor[0])
	if err != nil || createdAt.IsZero() {
		return nil, "", errInvalidRequest
	}
	return &createdAt, cursor[1], nil
}

func (h *auditHandlers) listOwnerFindings(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursor, err := pageQuery(r.URL.RawQuery, "state", "verdict", "severity")
	if err != nil {
		return err
	}
	params := auditstore.OwnerFindingListParams{OwnerID: principalUserID(r.Context()), Limit: pagelimit.FetchRows(limit)}
	if values, ok := query["state"]; ok {
		state := auditstore.FindingState(values[0])
		if !state.Valid() {
			return errInvalidRequest
		}
		params.State = &state
	}
	if values, ok := query["verdict"]; ok {
		if values[0] == "unreviewed" {
			params.Unreviewed = true
		} else {
			verdict := auditstore.AnalystVerdict(values[0])
			if verdict != auditstore.VerdictTruePositive && verdict != auditstore.VerdictFalsePositive {
				return errInvalidRequest
			}
			params.Verdict = &verdict
		}
	}
	if values, ok := query["severity"]; ok {
		severity := auditstore.FindingSeverity(values[0])
		if !severity.Valid() || params.Unreviewed {
			return errInvalidRequest
		}
		params.Severity = &severity
	}
	kind := "owner-findings:" + params.OwnerID + ":" + query.Get("state") + ":" + query.Get("verdict") + ":" + query.Get("severity")
	params.BeforeCreatedAt, params.BeforeFindingID, err = h.ownerListPosition(cursor, kind)
	if err != nil {
		return err
	}
	window, err := h.Audits.ListOwnerFindings(r.Context(), params)
	if err != nil {
		return err
	}
	items := window.Items
	items, page, err := paginate(h.cursors, items, limit, kind, func(last auditstore.ProjectFindingSummary) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID}
	})
	if err != nil {
		return err
	}
	if last := window.ContinueBefore; last != nil {
		cursor, err := h.cursors.encodePageCursor(kind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID)
		if err != nil {
			return err
		}
		page = pageInfoResponse{HasMore: true, NextCursor: &cursor}
	}
	if items == nil {
		items = []auditstore.ProjectFindingSummary{}
	}
	writeJSON(w, http.StatusOK, struct {
		Items []auditstore.ProjectFindingSummary `json:"items"`
		Page  pageInfoResponse                   `json:"page"`
	}{items, page})
	return nil
}
