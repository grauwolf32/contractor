package public

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

type findingPageResponse struct {
	Items []auditservice.Finding `json:"items"`
	Page  pageInfoResponse       `json:"page"`
}

type reviewPageResponse struct {
	Items []auditservice.ReviewRequest `json:"items"`
	Page  pageInfoResponse             `json:"page"`
}

type findingProvenancePageResponse struct {
	FindingRevision uint64                           `json:"findingRevision"`
	AuditRevision   uint64                           `json:"auditRevision"`
	Items           []auditservice.FindingProvenance `json:"items"`
	Page            pageInfoResponse                 `json:"page"`
}

type createFindingReviewRequest struct {
	ExpiresAt *time.Time `json:"expiresAt,omitempty"`
}

type decideFindingRequest struct {
	Verdict           auditservice.AnalystVerdict   `json:"verdict"`
	Severity          *auditservice.FindingSeverity `json:"severity,omitempty"`
	Rationale         string                        `json:"rationale"`
	DuplicateTargetID *string                       `json:"duplicateTargetId,omitempty"`
}

func (h *handler) listAuditFindings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursorValue, err := pageQuery(r.URL.RawQuery, "state", "verdict", "severity")
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditservice.FindingListParams{
		OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: limit + 1,
	}
	stateValue, verdictValue, severityValue := "", "", ""
	if values, ok := query["state"]; ok {
		value := auditservice.FindingState(values[0])
		if !value.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.State, stateValue = &value, values[0]
	}
	if values, ok := query["verdict"]; ok {
		verdictValue = values[0]
		if values[0] == "unreviewed" {
			params.Unreviewed = true
		} else {
			value := auditservice.AnalystVerdict(values[0])
			if value != auditservice.VerdictTruePositive && value != auditservice.VerdictFalsePositive {
				h.handleError(w, errInvalidRequest)
				return
			}
			params.Verdict = &value
		}
	}
	if values, ok := query["severity"]; ok {
		value := auditservice.FindingSeverity(values[0])
		if !value.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.Severity, severityValue = &value, values[0]
	}
	cursorKind := "audit-findings:" + params.AuditID + ":" + stateValue + ":" + verdictValue + ":" + severityValue
	cursor, err := h.decodePageCursor(cursorValue, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if len(cursor) != 0 {
		createdAt, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.AfterCreatedAt, params.AfterFindingID = &createdAt, cursor[1]
	}
	findings, err := h.dependencies.Audits.ListFindings(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(findings) > limit {
		findings = findings[:limit]
		last := findings[len(findings)-1]
		next, cursorErr := h.encodePageCursor(cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.FindingID)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	writeJSON(w, http.StatusOK, findingPageResponse{Items: findings, Page: page})
}

func (h *handler) getAuditFinding(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	finding, err := h.dependencies.Audits.GetFinding(
		r.Context(), principalUserID(r.Context()), r.PathValue("auditId"), r.PathValue("findingId"),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(finding.Revision, 10)))
	writeJSON(w, http.StatusOK, finding)
}

func (h *handler) createAuditFindingReview(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request createFindingReviewRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	key, revision, err := reviewMutationHeaders(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	requestID, err := h.dependencies.NewID("review_")
	if err != nil {
		h.handleError(w, err)
		return
	}
	auditID, findingID := r.PathValue("auditId"), r.PathValue("findingId")
	digest := reviewRequestDigest("create", auditID, findingID, revision, request)
	result, err := h.dependencies.Audits.CreateFindingReview(r.Context(), auditservice.CreateFindingReviewParams{
		OwnerID: principalUserID(r.Context()), AuditID: auditID, FindingID: findingID,
		ExpectedRevision: revision, RequestID: requestID, ExpiresAt: request.ExpiresAt,
		IdempotencyKey: key, RequestDigest: digest,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Request.Revision, 10)))
	writeJSON(w, http.StatusCreated, result.Request)
}

func (h *handler) listAuditReviews(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursorValue, err := pageQuery(r.URL.RawQuery, "finding", "state")
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditservice.ReviewListParams{
		OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"), Limit: limit + 1,
	}
	findingValue, stateValue := "", ""
	if values, ok := query["finding"]; ok {
		params.FindingID, findingValue = &values[0], values[0]
	}
	if values, ok := query["state"]; ok {
		value := auditservice.ReviewState(values[0])
		if !value.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.State, stateValue = &value, values[0]
	}
	cursorKind := "audit-reviews:" + params.AuditID + ":" + findingValue + ":" + stateValue
	cursor, err := h.decodePageCursor(cursorValue, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if len(cursor) != 0 {
		createdAt, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.AfterCreatedAt, params.AfterRequestID = &createdAt, cursor[1]
	}
	reviews, err := h.dependencies.Audits.ListReviews(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(reviews) > limit {
		reviews = reviews[:limit]
		last := reviews[len(reviews)-1]
		next, cursorErr := h.encodePageCursor(cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.RequestID)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	writeJSON(w, http.StatusOK, reviewPageResponse{Items: reviews, Page: page})
}

func (h *handler) decideAuditReview(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request decideFindingRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	key, revision, err := reviewMutationHeaders(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	decisionID, err := h.dependencies.NewID("decision_")
	if err != nil {
		h.handleError(w, err)
		return
	}
	auditID, requestID := r.PathValue("auditId"), r.PathValue("requestId")
	digest := reviewRequestDigest("decide", auditID, requestID, revision, request)
	result, err := h.dependencies.Audits.DecideFinding(r.Context(), auditservice.DecideFindingParams{
		OwnerID: principalUserID(r.Context()), AuditID: auditID, RequestID: requestID,
		ExpectedRequestRevision: revision, DecisionID: decisionID,
		Verdict: request.Verdict, Severity: request.Severity, Rationale: request.Rationale,
		DuplicateTargetID: request.DuplicateTargetID, IdempotencyKey: key, RequestDigest: digest,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(result.Request.Revision, 10)))
	writeJSON(w, http.StatusOK, result)
}

func (h *handler) listAuditFindingProvenance(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	query, limit, cursorValue, err := pageQuery(r.URL.RawQuery, "auditRevision", "findingRevision")
	if err != nil {
		h.handleError(w, err)
		return
	}
	ownerID, auditID, findingID := principalUserID(r.Context()), r.PathValue("auditId"), r.PathValue("findingId")
	audit, err := h.dependencies.Audits.Get(r.Context(), ownerID, auditID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	finding, err := h.dependencies.Audits.GetFinding(r.Context(), ownerID, auditID, findingID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	for name, actual := range map[string]uint64{
		"auditRevision": audit.Revision, "findingRevision": finding.Revision,
	} {
		if values, ok := query[name]; ok {
			expected, parseErr := strconv.ParseUint(values[0], 10, 64)
			if parseErr != nil || expected != actual {
				h.handleError(w, auditstore.ErrConflict)
				return
			}
		}
	}
	cursorKind := "audit-finding-provenance:" + auditID + ":" + findingID
	cursor, err := h.decodePageCursor(cursorValue, cursorKind, 4)
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditservice.ProvenanceListParams{
		OwnerID: ownerID, AuditID: auditID, FindingID: findingID, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		auditRevision, auditErr := strconv.ParseUint(cursor[0], 10, 64)
		findingRevision, findingErr := strconv.ParseUint(cursor[1], 10, 64)
		createdAt, timeErr := time.Parse(time.RFC3339Nano, cursor[2])
		if auditErr != nil || findingErr != nil || timeErr != nil ||
			auditRevision != audit.Revision || findingRevision != finding.Revision {
			h.handleError(w, auditstore.ErrConflict)
			return
		}
		params.AfterCreatedAt, params.AfterRecordID = &createdAt, cursor[3]
	}
	items, err := h.dependencies.Audits.ListFindingProvenance(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1]
		next, cursorErr := h.encodePageCursor(cursorKind,
			strconv.FormatUint(audit.Revision, 10), strconv.FormatUint(finding.Revision, 10),
			last.CreatedAt.UTC().Format(time.RFC3339Nano), last.RecordID)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	writeJSON(w, http.StatusOK, findingProvenancePageResponse{
		FindingRevision: finding.Revision, AuditRevision: audit.Revision, Items: items, Page: page,
	})
}

func reviewMutationHeaders(r *http.Request) (string, uint64, error) {
	key, err := requireIdempotencyKey(r)
	if err != nil {
		return "", 0, err
	}
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		return "", 0, fmt.Errorf("%w: review mutation requires one If-Match", errInvalidRequest)
	}
	revision, err := parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
	return key, revision, err
}

func reviewRequestDigest(kind string, identity ...any) string {
	encoded, _ := json.Marshal(struct {
		Schema   string `json:"schema"`
		Kind     string `json:"kind"`
		Identity []any  `json:"identity"`
	}{Schema: "contractor.audit.review-request.v1", Kind: kind, Identity: identity})
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:])
}
