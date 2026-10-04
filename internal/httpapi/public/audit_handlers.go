package public

import (
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
)

func (h *handler) listAuditProfiles(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, "audit-profiles", 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	profiles := h.dependencies.Audits.Profiles()
	sort.Slice(profiles, func(i, j int) bool {
		left := profiles[i].Profile.Ref.Name + "@" + profiles[i].Profile.Ref.Version
		right := profiles[j].Profile.Ref.Name + "@" + profiles[j].Profile.Ref.Version
		return left < right
	})
	items := make([]auditProfileResponse, 0, min(limit+1, len(profiles)))
	for _, profile := range profiles {
		selector := profile.Profile.Ref.Name + "@" + profile.Profile.Ref.Version
		if selector <= after {
			continue
		}
		items = append(items, auditProfileReadModel(profile, false))
		if len(items) == limit+1 {
			break
		}
	}
	items, page, err := paginate(h, items, limit, "audit-profiles", func(last auditProfileResponse) []string {
		return []string{last.Ref.Name + "@" + last.Ref.Version}
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, auditProfilePageResponse{Items: items, Page: page})
}

func (h *handler) getAuditProfile(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	selector := auditservice.ProfileSelector{Name: r.PathValue("name"), Version: r.PathValue("version")}
	if _, err := config.ParseSelector(selector.Name + "@" + selector.Version); err != nil {
		h.handleError(w, errInvalidRequest)
		return
	}
	profile, err := h.dependencies.Audits.Profile(selector)
	if err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(profile.Profile.Ref.Digest))
	writeJSON(w, http.StatusOK, auditProfileReadModel(profile, true))
}

func (h *handler) createAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	var request createAuditRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	projectID := r.PathValue("projectId")
	digest, labels, err := createAuditRequestDigest(projectID, request)
	if err != nil {
		h.handleError(w, err)
		return
	}
	auditID, err := h.dependencies.NewID("audit_")
	if err != nil {
		h.handleError(w, fmt.Errorf("generate Audit ID: %w", err))
		return
	}
	audit, created, err := h.dependencies.Audits.CreateDraft(r.Context(), auditservice.CreateDraftParams{
		AuditID: auditID, OwnerID: principalUserID(r.Context()), ProjectID: projectID,
		Profile: auditservice.ProfileSelector{Name: request.Profile.Name, Version: request.Profile.Version},
		Inputs:  request.Inputs, RuntimeLabels: labels, Scope: request.Scope,
		IdempotencyKey: key, RequestDigest: digest,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !created {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	response, err := auditReadModel(audit)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeAuditJSON(w, http.StatusCreated, response)
}

func (h *handler) listProjectAudits(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	projectID := r.PathValue("projectId")
	if _, err := h.dependencies.Projects.Get(r.Context(), principalUserID(r.Context()), projectID); err != nil {
		h.handleError(w, err)
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "state", "profile")
	if err != nil {
		h.handleError(w, err)
		return
	}
	var state *auditstore.AuditState
	if values, ok := query["state"]; ok {
		candidate := auditstore.AuditState(values[0])
		if !candidate.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		state = &candidate
	}
	var profileName, profileVersion *string
	profileSelector := ""
	if values, ok := query["profile"]; ok {
		selector, parseErr := config.ParseSelector(values[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		profileSelector = selector.String()
		profileName, profileVersion = &selector.ID, &selector.Version
	}
	cursorKind := "project-audits:" + projectID + ":" + pointerAuditState(state) + ":" + profileSelector
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditstore.ListParams{
		OwnerID: principalUserID(r.Context()), ProjectID: &projectID, State: state,
		ProfileName: profileName, ProfileVersion: profileVersion, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.BeforeCreatedAt, params.BeforeAuditID = &before, cursor[1]
	}
	audits, err := h.dependencies.Audits.List(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	audits, page, err := paginate(h, audits, limit, cursorKind, func(last auditstore.Audit) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.AuditID}
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	items := make([]auditResponse, 0, len(audits))
	for _, audit := range audits {
		item, modelErr := auditReadModel(audit)
		if modelErr != nil {
			h.handleError(w, modelErr)
			return
		}
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, auditPageResponse{Items: items, Page: page})
}

func (h *handler) getAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	audit, err := h.dependencies.Audits.Get(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"))
	if err != nil {
		h.handleError(w, err)
		return
	}
	response, err := auditReadModel(audit)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeAuditJSON(w, http.StatusOK, response)
}

func (h *handler) startAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	seconds, err := readAuditTimeLimit(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		h.handleError(w, fmt.Errorf("%w: Audit start requires one If-Match", errInvalidRequest))
		return
	}
	revision, err := parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
	if err != nil {
		h.handleError(w, err)
		return
	}
	auditID := r.PathValue("auditId")
	digest := auditRequestDigest("", auditID, revision, seconds)
	credentialUser := runtimeCredentialUser(r.Context())
	started, err := h.dependencies.Audits.Start(r.Context(), auditservice.StartParams{
		OwnerID: credentialUser.UserID, OperationsPrincipal: credentialUser.Operations,
		AuditID: auditID, ExpectedRevision: revision,
		IdempotencyKey: key, RequestDigest: digest, DeadlineSeconds: seconds,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if started.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	audit, err := auditReadModel(started.Audit)
	if err != nil {
		h.handleError(w, err)
		return
	}
	items := make([]auditItemResponse, len(started.Items))
	for index := range started.Items {
		items[index] = auditItemReadModel(started.Items[index])
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(audit.Revision, 10)))
	round := auditRoundReadModel(started.Round)
	writeJSON(w, http.StatusOK, auditStartResponse{
		Audit: audit, Round: &round, Items: items,
	})
}

func (h *handler) pauseAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "pause", http.StatusOK)
}

func (h *handler) resumeAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "resume", http.StatusOK)
}

func (h *handler) cancelAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "cancel", http.StatusAccepted)
}

func (h *handler) deleteAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "delete", http.StatusAccepted)
}

func (h *handler) mutateAudit(w http.ResponseWriter, r *http.Request, action string, status int) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var seconds *int
	var err error
	if action == "resume" {
		seconds, err = readAuditTimeLimit(w, r)
	} else {
		err = requireEmptyBody(w, r)
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		h.handleError(w, fmt.Errorf("%w: Audit mutation requires one If-Match", errInvalidRequest))
		return
	}
	revision, err := parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditservice.MutationParams{
		OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"),
		ExpectedRevision: revision, IdempotencyKey: key,
		RequestDigest:   auditRequestDigest(action, r.PathValue("auditId"), revision, seconds),
		DeadlineSeconds: seconds,
	}
	var result auditservice.MutationResult
	switch action {
	case "pause":
		result, err = h.dependencies.Audits.Pause(r.Context(), params)
	case "resume":
		result, err = h.dependencies.Audits.Resume(r.Context(), params)
	case "cancel":
		result, err = h.dependencies.Audits.Cancel(r.Context(), params)
	case "delete":
		result, err = h.dependencies.Audits.Delete(r.Context(), params)
	default:
		err = errInvalidRequest
	}
	if err != nil {
		h.handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	response, err := auditReadModel(result.Audit)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeAuditJSON(w, status, response)
}

func (h *handler) listAuditItems(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "round", "state", "subject")
	if err != nil {
		h.handleError(w, err)
		return
	}
	var roundID, subject *string
	if values, ok := query["round"]; ok {
		roundID = &values[0]
	}
	if values, ok := query["subject"]; ok {
		subject = &values[0]
	}
	var state *auditstore.ItemState
	if values, ok := query["state"]; ok {
		candidate := auditstore.ItemState(values[0])
		if !candidate.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		state = &candidate
	}
	auditID := r.PathValue("auditId")
	cursorKind := "audit-items:" + auditID + ":" + pointerString(roundID) + ":" + pointerItemState(state) + ":" + pointerString(subject)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 3)
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := auditstore.ListItemsParams{
		OwnerID: principalUserID(r.Context()), AuditID: auditID, RoundID: roundID,
		State: state, SubjectKey: subject, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		roundOrdinal, roundErr := strconv.Atoi(cursor[0])
		itemOrdinal, itemErr := strconv.Atoi(cursor[1])
		if roundErr != nil || itemErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.AfterRoundOrdinal, params.AfterItemOrdinal, params.AfterItemID = &roundOrdinal, &itemOrdinal, cursor[2]
	}
	items, err := h.dependencies.Audits.ListItems(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1]
		round, roundErr := h.dependencies.Audits.GetRound(r.Context(), params.OwnerID, auditID, last.RoundID)
		if roundErr != nil {
			h.handleError(w, roundErr)
			return
		}
		next, cursorErr := h.encodePageCursor(
			cursorKind, strconv.Itoa(round.Ordinal), strconv.Itoa(last.Ordinal), last.ItemID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	attempts := map[string][]auditstore.ItemAttempt{}
	if len(items) != 0 {
		itemIDs := make([]string, len(items))
		for index := range items {
			itemIDs[index] = items[index].ItemID
		}
		attempts, err = h.dependencies.Audits.ListItemAttempts(
			r.Context(), params.OwnerID, auditID, itemIDs,
		)
		if err != nil {
			h.handleError(w, err)
			return
		}
	}
	response := make([]auditItemResponse, len(items))
	for index := range items {
		response[index] = auditItemReadModel(items[index])
		response[index].Attempts = attempts[items[index].ItemID]
	}
	writeJSON(w, http.StatusOK, auditItemPageResponse{Items: response, Page: page})
}

func (h *handler) listAuditCoverage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "round")
	if err != nil {
		h.handleError(w, err)
		return
	}
	ownerID, auditID := principalUserID(r.Context()), r.PathValue("auditId")
	audit, err := h.dependencies.Audits.Get(r.Context(), ownerID, auditID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	roundID := ""
	if values, ok := query["round"]; ok {
		roundID = values[0]
	} else if audit.CurrentRoundID != nil {
		roundID = *audit.CurrentRoundID
	}
	if roundID == "" {
		writeJSON(w, http.StatusOK, auditCoveragePageResponse{Items: []auditCoverageResponse{}, Page: pageInfoResponse{}})
		return
	}
	cursorKind := "audit-coverage:" + auditID + ":" + roundID
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := -1
	if len(cursor) != 0 {
		after, err = strconv.Atoi(cursor[0])
		if err != nil || after < 0 {
			h.handleError(w, errInvalidRequest)
			return
		}
	}
	coverage, err := h.dependencies.Audits.ListCoverage(r.Context(), ownerID, auditID, roundID, after, limit+1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	coverage, page, err := paginate(h, coverage, limit, cursorKind, func(last auditstore.CoverageRow) []string {
		return []string{strconv.Itoa(last.Ordinal)}
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	items := make([]auditCoverageResponse, len(coverage))
	for index, row := range coverage {
		items[index] = auditCoverageResponse{
			RoundID: row.RoundID, ItemID: row.ItemID, Ordinal: row.Ordinal,
			ItemKey: row.ItemKey, SubjectKey: row.SubjectKey, Coverage: row.Coverage,
			Result: row.Result, UpdatedAt: row.UpdatedAt, Details: row.Details,
		}
	}
	writeJSON(w, http.StatusOK, auditCoveragePageResponse{Items: items, Page: page})
}

func (h *handler) getAuditReport(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	projection, err := h.dependencies.Audits.GetReport(
		r.Context(), principalUserID(r.Context()), r.PathValue("auditId"),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, auditReportResponse{
		Status: projection.Status, Review: projection.Review, MachineArtifact: projection.MachineArtifact,
		SummaryArtifact: projection.SummaryArtifact, Machine: projection.Machine,
		Summary: projection.Summary,
	})
}
