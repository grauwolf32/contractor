package public

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

type findingCorrelationCursor struct {
	Version  int                                      `json:"v" strictjson:"required"`
	Kind     string                                   `json:"k" strictjson:"required"`
	Position *auditservice.FindingCorrelationPosition `json:"p" strictjson:"required"`
}

func findingCorrelationQuery(r *http.Request, extra ...string) (url.Values, int, string, error) {
	query, err := exactQuery(r.URL.RawQuery, append([]string{"limit", "cursor"}, extra...)...)
	if err != nil {
		return nil, 0, "", err
	}
	limit := auditstore.FindingComparisonDefaultLimit
	if values, ok := query["limit"]; ok {
		limit, err = strconv.Atoi(values[0])
		if err != nil || limit < 1 || limit > auditstore.FindingComparisonMaximumLimit {
			return nil, 0, "", errInvalidRequest
		}
	}
	cursor := query.Get("cursor")
	if values, ok := query["cursor"]; ok && (values[0] == "" || len(cursor) > auditservice.FindingCorrelationMaximumCursorBytes) {
		return nil, 0, "", errInvalidRequest
	}
	return query, limit, cursor, nil
}
func findingCorrelationKind(values ...any) string {
	raw, _ := json.Marshal(values)
	return "finding-correlation/v1:" + string(raw)
}
func (c pageCursors) decodeFindingCorrelation(encoded, kind string) (*auditservice.FindingCorrelationPosition, error) {
	if encoded == "" {
		return nil, nil
	}
	if len(encoded) > auditservice.FindingCorrelationMaximumCursorBytes {
		return nil, errInvalidRequest
	}
	raw, ok := c.open("finding-correlation/v1", encoded)
	if !ok {
		return nil, errInvalidRequest
	}
	var value findingCorrelationCursor
	if err := strictjson.Decode(raw, &value); err != nil || value.Version != 1 || value.Kind != pageCursorKind(kind) || value.Position == nil {
		return nil, errInvalidRequest
	}
	return value.Position, nil
}
func (c pageCursors) encodeFindingCorrelation(kind string, position *auditservice.FindingCorrelationPosition) (*string, error) {
	if position == nil {
		return nil, nil
	}
	raw, err := json.Marshal(findingCorrelationCursor{Version: 1, Kind: pageCursorKind(kind), Position: position})
	if err != nil {
		return nil, err
	}
	value := c.seal("finding-correlation/v1", raw)
	if len(value) > auditservice.FindingCorrelationMaximumCursorBytes {
		return nil, fmt.Errorf("finding correlation cursor exceeds bound")
	}
	return &value, nil
}
func correlationQueryID(query url.Values, key string, required bool) bool {
	values, ok := query[key]
	if !ok {
		return !required
	}
	return contracts.ValidResourceID(values[0])
}

func (h *auditHandlers) listFindingDuplicateCandidates(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	query, limit, cursor, err := findingCorrelationQuery(r, "scope", "targetAuditId", "sourceReceiptId")
	if err != nil {
		handleError(w, err)
		return
	}
	if !correlationQueryID(query, "targetAuditId", false) || !correlationQueryID(query, "sourceReceiptId", false) {
		handleError(w, errInvalidRequest)
		return
	}
	scope := query.Get("scope")
	if _, ok := query["scope"]; !ok {
		scope = "project"
	}
	p := auditservice.FindingCandidatesParams{OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), FindingID: r.PathValue("findingId"), Scope: scope, TargetAuditID: query.Get("targetAuditId"), SourceReceiptID: query.Get("sourceReceiptId"), Limit: limit}
	kind := findingCorrelationKind("candidates", p.OwnerID, p.AuditID, p.FindingID, p.Scope, p.TargetAuditID, p.SourceReceiptID, p.Limit)
	p.Cursor, err = h.cursors.decodeFindingCorrelation(cursor, kind)
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.Audits.FindingCandidates(r.Context(), p)
	if err != nil {
		handleError(w, err)
		return
	}
	result.Page.NextCursor, err = h.cursors.encodeFindingCorrelation(kind, result.Next)
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
func (h *auditHandlers) listProjectFindingComparisons(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	query, limit, cursor, err := findingCorrelationQuery(r, "leftAuditId", "rightAuditId")
	if err != nil {
		handleError(w, err)
		return
	}
	if !correlationQueryID(query, "leftAuditId", true) || !correlationQueryID(query, "rightAuditId", true) {
		handleError(w, errInvalidRequest)
		return
	}
	p := auditservice.FindingComparisonParams{OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"), LeftAuditID: query.Get("leftAuditId"), RightAuditID: query.Get("rightAuditId"), Limit: limit}
	kind := findingCorrelationKind("comparison", p.OwnerID, p.ProjectID, p.LeftAuditID, p.RightAuditID, p.Limit)
	p.Cursor, err = h.cursors.decodeFindingCorrelation(cursor, kind)
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.Audits.CompareFindings(r.Context(), p)
	if err != nil {
		handleError(w, err)
		return
	}
	result.Page.NextCursor, err = h.cursors.encodeFindingCorrelation(kind, result.Next)
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
func (h *auditHandlers) listFindingCorrelationMembers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	query, limit, cursor, err := findingCorrelationQuery(r, "auditId", "leftAuditId", "rightAuditId")
	if err != nil {
		handleError(w, err)
		return
	}
	for _, key := range []string{"auditId", "leftAuditId", "rightAuditId"} {
		if !correlationQueryID(query, key, false) {
			handleError(w, errInvalidRequest)
			return
		}
	}
	ids := []string{}
	if id, ok := query["auditId"]; ok {
		if query.Has("leftAuditId") || query.Has("rightAuditId") {
			handleError(w, errInvalidRequest)
			return
		}
		ids = append(ids, id[0])
	} else if query.Has("leftAuditId") || query.Has("rightAuditId") {
		if !query.Has("leftAuditId") || !query.Has("rightAuditId") {
			handleError(w, errInvalidRequest)
			return
		}
		ids = append(ids, query.Get("leftAuditId"), query.Get("rightAuditId"))
	}
	p := auditservice.FindingMembersParams{OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"), GroupKey: r.PathValue("groupKey"), AuditIDs: ids, Limit: limit}
	kind := findingCorrelationKind("members", p.OwnerID, p.ProjectID, p.GroupKey, p.AuditIDs, p.Limit)
	p.Cursor, err = h.cursors.decodeFindingCorrelation(cursor, kind)
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.Audits.FindingGroupMembers(r.Context(), p)
	if err != nil {
		handleError(w, err)
		return
	}
	result.Page.NextCursor, err = h.cursors.encodeFindingCorrelation(kind, result.Next)
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
