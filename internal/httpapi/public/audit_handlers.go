package public

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

type AuditManagement interface {
	FindingCandidates(context.Context, auditservice.FindingCandidatesParams) (auditservice.FindingCandidatesPage, error)
	CompareFindings(context.Context, auditservice.FindingComparisonParams) (auditservice.FindingComparisonPage, error)
	FindingGroupMembers(context.Context, auditservice.FindingMembersParams) (auditservice.FindingMembersPage, error)
	ListEventsPage(context.Context, auditstore.EventListParams) (auditservice.EventPage, error)
	ListOwnerFindings(context.Context, auditstore.OwnerFindingListParams) ([]auditservice.Finding, error)
	ListOwnerReviews(context.Context, auditstore.OwnerReviewListParams) ([]auditstore.ReviewRequest, error)
	GetWorkspace(context.Context, string, string) (auditstore.WorkspaceSummary, error)
	ListFindingsPage(context.Context, auditstore.FindingListParams) (auditservice.FindingPage, error)
	ListReviewsPage(context.Context, auditstore.ReviewListParams) (auditservice.ReviewPage, error)
	Profiles() []auditservice.ProfileProjection
	Profile(auditservice.ProfileSelector) (auditservice.ProfileProjection, error)
	Standards(context.Context, string) ([]auditstandards.PackageProjection, error)
	Standard(context.Context, string, auditstandards.Reference) (auditstandards.PackageProjection, error)
	CreateDraft(context.Context, auditservice.CreateDraftParams) (auditstore.Audit, bool, error)
	Start(context.Context, auditservice.StartParams) (auditservice.StartedAudit, error)
	Pause(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Resume(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Cancel(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Delete(context.Context, auditservice.MutationParams) (auditservice.MutationResult, error)
	Get(context.Context, string, string) (auditstore.Audit, error)
	Preparation(context.Context, string, string) (map[string]auditservice.PreparationRoleProjection, error)
	ReadArtifact(context.Context, string, string, string, string) (artifacts.ReadResult, error)
	GeneratedInventory(context.Context, string, string, uint64) (*auditservice.GeneratedInventoryProjection, error)
	List(context.Context, auditstore.ListParams) ([]auditstore.Audit, error)
	ListItems(context.Context, auditstore.ListItemsParams) ([]auditstore.Item, error)
	ListItemAttempts(context.Context, string, string, []string) (map[string][]auditstore.ItemAttempt, error)
	GetRound(context.Context, string, string, string) (auditstore.Round, error)
	ListCoverage(context.Context, string, string, string, int, int, *uint64) (auditservice.CoveragePage, error)
	GetReport(context.Context, string, string) (auditservice.ReportProjection, error)
	GetFinding(context.Context, string, string, string) (auditservice.Finding, error)
	CreateFindingReview(context.Context, auditservice.CreateFindingReviewParams) (auditservice.FindingReviewResult, error)
	DecideFinding(context.Context, auditservice.DecideFindingParams) (auditservice.FindingDecisionResult, error)
	DecideActionReview(context.Context, auditservice.DecideActionReviewParams) (auditservice.ActionReviewDecisionResult, error)
	GetReview(context.Context, string, string, string) (auditstore.ReviewRequest, error)
	ListFindingProvenance(context.Context, auditstore.ProvenanceListParams) ([]auditservice.FindingProvenance, error)
}

// AuditDependencies serve AuditProfiles, AuditStandards, Audits, their
// items, coverage, events, findings, reviews and reports, and the owner-wide
// Audit, finding and review lists.
type AuditDependencies struct {
	Audits   AuditManagement
	Projects ProjectReader
	NewID    func(string) (string, error)
}

type auditHandlers struct {
	AuditDependencies
	cursors pageCursors
}

func (h *auditHandlers) listAuditProfiles(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "audit-profiles", 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	profiles := h.Audits.Profiles()
	sort.Slice(profiles, func(i, j int) bool {
		left := profiles[i].Profile.Ref.Name + "@" + profiles[i].Profile.Ref.Version
		right := profiles[j].Profile.Ref.Name + "@" + profiles[j].Profile.Ref.Version
		return left < right
	})
	items := make([]auditProfileResponse, 0, min(pagelimit.FetchRows(limit), len(profiles)))
	for _, profile := range profiles {
		selector := profile.Profile.Ref.Name + "@" + profile.Profile.Ref.Version
		if selector <= after {
			continue
		}
		items = append(items, auditProfileReadModel(profile, false))
		if len(items) == pagelimit.FetchRows(limit) {
			break
		}
	}
	items, page, err := paginate(h.cursors, items, limit, "audit-profiles", func(last auditProfileResponse) []string {
		return []string{last.Ref.Name + "@" + last.Ref.Version}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, auditProfilePageResponse{Items: items, Page: page})
}

func (h *auditHandlers) getAuditProfile(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	selector := auditservice.ProfileSelector{Name: r.PathValue("name"), Version: r.PathValue("version")}
	if _, err := resolved.ParseSelector(selector.Name + "@" + selector.Version); err != nil {
		handleError(w, errInvalidRequest)
		return
	}
	profile, err := h.Audits.Profile(selector)
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(profile.Profile.Ref.Digest))
	writeJSON(w, http.StatusOK, auditProfileReadModel(profile, true))
}

func (h *auditHandlers) createAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	var request createAuditRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	projectID := r.PathValue("projectId")
	digest, labels, err := createAuditRequestDigest(projectID, request)
	if err != nil {
		handleError(w, err)
		return
	}
	auditID, err := h.NewID("audit_")
	if err != nil {
		handleError(w, fmt.Errorf("generate Audit ID: %w", err))
		return
	}
	audit, created, err := h.Audits.CreateDraft(r.Context(), auditservice.CreateDraftParams{
		AuditID: auditID, OwnerID: principalUserID(r.Context()), ProjectID: projectID,
		Profile: auditservice.ProfileSelector{Name: request.Profile.Name, Version: request.Profile.Version},
		Inputs:  request.Inputs, RuntimeLabels: labels, Scope: request.Scope,
		IdempotencyKey: key, RequestDigest: digest,
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if !created {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	response, err := h.auditDetailReadModel(r, audit)
	if err != nil {
		handleError(w, err)
		return
	}
	writeAuditJSON(w, http.StatusCreated, response)
}

func (h *auditHandlers) listProjectAudits(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	projectID := r.PathValue("projectId")
	if projectID != "" {
		if _, err := h.Projects.Get(r.Context(), principalUserID(r.Context()), projectID); err != nil {
			handleError(w, err)
			return
		}
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "state", "profile")
	if err != nil {
		handleError(w, err)
		return
	}
	// The owner-wide list accepts a comma-separated state set; a Project list
	// filters by one state.
	var states []auditstore.AuditState
	if values, ok := query["state"]; ok {
		if projectID == "" {
			states, err = parseAuditStates(values[0])
			if err != nil {
				handleError(w, err)
				return
			}
		} else {
			candidate := auditstore.AuditState(values[0])
			if !candidate.Valid() {
				handleError(w, errInvalidRequest)
				return
			}
			states = []auditstore.AuditState{candidate}
		}
	}
	var profileName, profileVersion *string
	profileSelector := ""
	if values, ok := query["profile"]; ok {
		selector, parseErr := resolved.ParseSelector(values[0])
		if parseErr != nil {
			handleError(w, errInvalidRequest)
			return
		}
		profileSelector = selector.String()
		profileName, profileVersion = &selector.ID, &selector.Version
	}
	cursorKind := "project-audits:" + projectID + ":" + auditStateKey(states) + ":" + profileSelector
	if projectID == "" {
		cursorKind = "owner-audits:" + principalUserID(r.Context()) + ":" + auditStateKey(states) + ":" + profileSelector
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		handleError(w, err)
		return
	}
	params := auditstore.ListParams{
		OwnerID: principalUserID(r.Context()), States: states, VisibleProjects: projectID == "",
		ProfileName: profileName, ProfileVersion: profileVersion, Limit: pagelimit.FetchRows(limit),
	}
	if projectID != "" {
		params.ProjectID = &projectID
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			handleError(w, errInvalidRequest)
			return
		}
		params.BeforeCreatedAt, params.BeforeAuditID = &before, cursor[1]
	}
	audits, err := h.Audits.List(r.Context(), params)
	if err != nil {
		handleError(w, err)
		return
	}
	audits, page, err := paginate(h.cursors, audits, limit, cursorKind, func(last auditstore.Audit) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.AuditID}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	items := make([]auditResponse, 0, len(audits))
	for _, audit := range audits {
		item, modelErr := auditReadModel(audit)
		if modelErr == nil && (audit.Phase == auditdomain.AuditPhasePreparing || audit.Phase == auditdomain.AuditPhaseInventory) {
			item, modelErr = h.auditDetailReadModel(r, audit)
		}
		if modelErr != nil {
			handleError(w, modelErr)
			return
		}
		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, auditPageResponse{Items: items, Page: page})
}

func (h *auditHandlers) getAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	audit, err := h.Audits.Get(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"))
	if err != nil {
		handleError(w, err)
		return
	}
	response, err := h.auditDetailReadModel(r, audit)
	if err != nil {
		handleError(w, err)
		return
	}
	writeAuditJSON(w, http.StatusOK, response)
}

func (h *auditHandlers) startAudit(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	seconds, err := readAuditTimeLimit(w, r)
	if err != nil {
		handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		handleError(w, fmt.Errorf("%w: Audit start requires one If-Match", errInvalidRequest))
		return
	}
	revision, err := parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
	if err != nil {
		handleError(w, err)
		return
	}
	auditID := r.PathValue("auditId")
	digest := auditRequestDigest("start", auditID, revision, seconds)
	credentialUser := runtimeCredentialUser(r.Context())
	started, err := h.Audits.Start(r.Context(), auditservice.StartParams{
		OwnerID: credentialUser.UserID, OperationsPrincipal: credentialUser.Operations,
		AuditID: auditID, ExpectedRevision: revision,
		IdempotencyKey: key, RequestDigest: digest, DeadlineSeconds: seconds,
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if started.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	audit, err := h.auditDetailReadModel(r, started.Audit)
	if err != nil {
		handleError(w, err)
		return
	}
	items := make([]auditItemResponse, len(started.Items))
	for index := range started.Items {
		items[index] = auditItemReadModel(started.Items[index])
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(audit.Revision, 10)))
	var round *auditRoundResponse
	if started.Audit.CurrentRoundID != nil {
		value := auditRoundReadModel(started.Round)
		round = &value
	}
	writeJSON(w, http.StatusOK, auditStartResponse{
		Audit: audit, Round: round, Items: items,
	})
}

func (h *auditHandlers) pauseAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "pause", http.StatusOK)
}

func (h *auditHandlers) resumeAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "resume", http.StatusOK)
}

func (h *auditHandlers) cancelAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "cancel", http.StatusAccepted)
}

func (h *auditHandlers) deleteAudit(w http.ResponseWriter, r *http.Request) {
	h.mutateAudit(w, r, "delete", http.StatusAccepted)
}

func (h *auditHandlers) mutateAudit(w http.ResponseWriter, r *http.Request, action string, status int) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
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
		handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		handleError(w, fmt.Errorf("%w: Audit mutation requires one If-Match", errInvalidRequest))
		return
	}
	revision, err := parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
	if err != nil {
		handleError(w, err)
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
		result, err = h.Audits.Pause(r.Context(), params)
	case "resume":
		result, err = h.Audits.Resume(r.Context(), params)
	case "cancel":
		result, err = h.Audits.Cancel(r.Context(), params)
	case "delete":
		result, err = h.Audits.Delete(r.Context(), params)
	default:
		err = errInvalidRequest
	}
	if err != nil {
		handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	response, err := h.auditDetailReadModel(r, result.Audit)
	if err != nil {
		handleError(w, err)
		return
	}
	writeAuditJSON(w, status, response)
}

func (h *auditHandlers) listAuditItems(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "round", "state", "subject")
	if err != nil {
		handleError(w, err)
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
			handleError(w, errInvalidRequest)
			return
		}
		state = &candidate
	}
	auditID := r.PathValue("auditId")
	cursorKind := "audit-items:" + auditID + ":" + pointerString(roundID) + ":" + pointerItemState(state) + ":" + pointerString(subject)
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 3)
	if err != nil {
		handleError(w, err)
		return
	}
	params := auditstore.ListItemsParams{
		OwnerID: principalUserID(r.Context()), AuditID: auditID, RoundID: roundID,
		State: state, SubjectKey: subject, Limit: pagelimit.FetchRows(limit),
	}
	if len(cursor) != 0 {
		roundOrdinal, roundErr := strconv.Atoi(cursor[0])
		itemOrdinal, itemErr := strconv.Atoi(cursor[1])
		if roundErr != nil || itemErr != nil {
			handleError(w, errInvalidRequest)
			return
		}
		params.AfterRoundOrdinal, params.AfterItemOrdinal, params.AfterItemID = &roundOrdinal, &itemOrdinal, cursor[2]
	}
	items, err := h.Audits.ListItems(r.Context(), params)
	if err != nil {
		handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1]
		round, roundErr := h.Audits.GetRound(r.Context(), params.OwnerID, auditID, last.RoundID)
		if roundErr != nil {
			handleError(w, roundErr)
			return
		}
		next, cursorErr := h.cursors.encodePageCursor(
			cursorKind, strconv.Itoa(round.Ordinal), strconv.Itoa(last.Ordinal), last.ItemID,
		)
		if cursorErr != nil {
			handleError(w, cursorErr)
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
		attempts, err = h.Audits.ListItemAttempts(
			r.Context(), params.OwnerID, auditID, itemIDs,
		)
		if err != nil {
			handleError(w, err)
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

func (h *auditHandlers) listAuditCoverage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "round", "auditRevision")
	if err != nil {
		handleError(w, err)
		return
	}
	ownerID, auditID := principalUserID(r.Context()), r.PathValue("auditId")
	audit, err := h.Audits.Get(r.Context(), ownerID, auditID)
	if err != nil {
		handleError(w, err)
		return
	}
	roundID := ""
	if values, ok := query["round"]; ok {
		roundID = values[0]
	} else if audit.CurrentRoundID != nil {
		roundID = *audit.CurrentRoundID
	}
	cursorKind := "audit-coverage:" + auditID + ":" + roundID
	cursor, revision, err := h.decodeAuditPageCursor(encodedCursor, cursorKind, query.Get("auditRevision"), 2)
	if err != nil {
		handleError(w, err)
		return
	}
	// The page, and the current Round it was chosen from, belong to one
	// revision: the pinned one, or the one just read.
	if revision == nil {
		revision = &audit.Revision
	} else if *revision != audit.Revision {
		handleError(w, auditstore.ErrConflict)
		return
	}
	if roundID == "" {
		writeJSON(w, http.StatusOK, auditCoveragePageResponse{
			Items: []auditCoverageResponse{}, Page: pageInfoResponse{}, AuditRevision: audit.Revision,
		})
		return
	}
	after := -1
	if len(cursor) != 0 {
		after, err = strconv.Atoi(cursor[0])
		if err != nil || after < 0 {
			handleError(w, errInvalidRequest)
			return
		}
	}
	result, err := h.Audits.ListCoverage(
		r.Context(), ownerID, auditID, roundID, after, pagelimit.FetchRows(limit), revision,
	)
	if err != nil {
		handleError(w, err)
		return
	}
	coverage, page, err := paginate(h.cursors, result.Rows, limit, cursorKind, func(last auditstore.CoverageRow) []string {
		return []string{strconv.Itoa(last.Ordinal), strconv.FormatUint(result.AuditRevision, 10)}
	})
	if err != nil {
		handleError(w, err)
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
	writeJSON(w, http.StatusOK, auditCoveragePageResponse{Items: items, Page: page, AuditRevision: result.AuditRevision})
}

func (h *auditHandlers) getAuditReport(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	projection, err := h.Audits.GetReport(
		r.Context(), principalUserID(r.Context()), r.PathValue("auditId"),
	)
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, auditReportResponse{
		Status: projection.Status, Review: projection.Review, MachineArtifact: projection.MachineArtifact,
		SummaryArtifact: projection.SummaryArtifact, Machine: projection.Machine,
		Summary: projection.Summary,
	})
}
