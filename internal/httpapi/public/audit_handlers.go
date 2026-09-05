package public

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type createAuditRequest struct {
	Profile       auditProfileSelectorRequest      `json:"profile"`
	Inputs        map[string]contracts.ArtifactRef `json:"inputs"`
	RuntimeLabels []string                         `json:"runtimeLabels,omitempty"`
	Scope         auditservice.Scope               `json:"scope"`
}

type auditProfileSelectorRequest struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type auditProfileResponse struct {
	Ref                     config.AuditProfileRef                  `json:"ref"`
	Mode                    config.AuditProfileMode                 `json:"mode"`
	Standards               []config.AuditStandardRef               `json:"standards"`
	Inputs                  map[string]config.AuditProfileInput     `json:"inputs"`
	Inventory               config.AuditInventory                   `json:"inventory"`
	Workflows               map[string]auditProfileWorkflowResponse `json:"workflows,omitempty"`
	Execution               config.AuditExecutionPolicy             `json:"execution"`
	Interaction             config.AuditInteractionPolicy           `json:"interaction"`
	ServerCompatible        bool                                    `json:"serverCompatible"`
	RequiresInputValidation bool                                    `json:"requiresInputValidation"`
	CompatibilityReasons    []auditservice.CompatibilityReason      `json:"compatibilityReasons"`
}

type auditProfileWorkflowResponse struct {
	Workflow   config.WorkflowRef                              `json:"workflow"`
	Inputs     map[string]config.AuditWorkflowInputMapping     `json:"inputs"`
	Parameters map[string]config.AuditWorkflowParameterMapping `json:"parameters"`
	Outputs    map[string]string                               `json:"outputs"`
}

type auditProfilePageResponse struct {
	Items []auditProfileResponse `json:"items"`
	Page  pageInfoResponse       `json:"page"`
}

type auditRuntimeSnapshotResponse struct {
	Default runtimeconfig.PinnedLabel   `json:"default"`
	Labels  []runtimeconfig.PinnedLabel `json:"labels"`
}

type auditBaselineResponse struct {
	Inputs            map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope             auditservice.Scope                  `json:"scope"`
	RuntimeLabels     []string                            `json:"runtimeLabels"`
	RuntimeConfig     auditRuntimeSnapshotResponse        `json:"runtimeConfig"`
	Skills            []auditSkillResponse                `json:"skills"`
	ProjectHTTPTarget *contracts.HTTPOriginTargetRef      `json:"projectHttpTarget,omitempty"`
	Inventory         auditBaselineInventoryResponse      `json:"inventory"`
}

type auditSkillResponse struct {
	Name         string                `json:"name"`
	Source       contracts.ArtifactRef `json:"source"`
	SourceDigest string                `json:"sourceDigest"`
	SourceSize   int64                 `json:"sourceSize"`
}

type auditBaselineInventoryResponse struct {
	SourceContentDigest      string                   `json:"sourceContentDigest"`
	CanonicalInventoryDigest string                   `json:"canonicalInventoryDigest"`
	Gaps                     []string                 `json:"gaps"`
	Worklist                 auditstore.ExactArtifact `json:"worklist"`
}

type auditProfileIdentityResponse struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type auditStopReasonResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type auditResponse struct {
	AuditID               string                              `json:"auditId"`
	ProjectID             string                              `json:"projectId"`
	Profile               auditProfileIdentityResponse        `json:"profile"`
	Inputs                map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope                 auditservice.Scope                  `json:"scope"`
	RuntimeLabels         []string                            `json:"runtimeLabels"`
	Baseline              *auditBaselineResponse              `json:"baseline,omitempty"`
	State                 auditstore.AuditState               `json:"state"`
	Revision              uint64                              `json:"revision"`
	CurrentRoundID        *string                             `json:"currentRoundId,omitempty"`
	DispatchState         auditstore.DispatchState            `json:"dispatchState"`
	HoldState             auditstore.HoldState                `json:"holdState"`
	DeadlineAt            *time.Time                          `json:"deadlineAt,omitempty"`
	Limits                auditstore.Limits                   `json:"limits"`
	ReservedRunCount      int                                 `json:"reservedRunCount"`
	SubmittedRunCount     int                                 `json:"submittedRunCount"`
	OutstandingRunCount   int                                 `json:"outstandingRunCount"`
	RetainedEvidenceBytes int64                               `json:"retainedEvidenceBytes"`
	EventSequence         uint64                              `json:"eventSequence"`
	StopReason            *auditStopReasonResponse            `json:"stopReason,omitempty"`
	CreatedAt             time.Time                           `json:"createdAt"`
	UpdatedAt             time.Time                           `json:"updatedAt"`
	StartedAt             *time.Time                          `json:"startedAt,omitempty"`
	FinishedAt            *time.Time                          `json:"finishedAt,omitempty"`
}

type auditPageResponse struct {
	Items []auditResponse  `json:"items"`
	Page  pageInfoResponse `json:"page"`
}

type auditRoundResponse struct {
	RoundID           string                   `json:"roundId"`
	Ordinal           int                      `json:"ordinal"`
	Manifest          auditstore.ExactArtifact `json:"manifest"`
	State             auditstore.RoundState    `json:"state"`
	ExpectedItemCount int                      `json:"expectedItemCount"`
	Revision          uint64                   `json:"revision"`
	CreatedAt         time.Time                `json:"createdAt"`
	UpdatedAt         time.Time                `json:"updatedAt"`
}

type auditItemResponse struct {
	ItemID              string                       `json:"itemId"`
	RoundID             string                       `json:"roundId"`
	ItemKey             string                       `json:"itemKey"`
	Ordinal             int                          `json:"ordinal"`
	Kind                string                       `json:"kind"`
	SubjectKey          string                       `json:"subjectKey"`
	Task                auditstore.ExactArtifact     `json:"task"`
	WorkflowRole        string                       `json:"workflowRole"`
	State               auditstore.ItemState         `json:"state"`
	FinalDisposition    *auditstore.FinalDisposition `json:"finalDisposition,omitempty"`
	AcceptedResult      *auditstore.ExactArtifact    `json:"acceptedResult,omitempty"`
	LastExecutionItemID *string                      `json:"lastExecutionItemId,omitempty"`
	CreatedAt           time.Time                    `json:"createdAt"`
	UpdatedAt           time.Time                    `json:"updatedAt"`
}

type auditItemPageResponse struct {
	Items []auditItemResponse `json:"items"`
	Page  pageInfoResponse    `json:"page"`
}

type auditCoverageResponse struct {
	RoundID    string                    `json:"roundId"`
	ItemID     string                    `json:"itemId"`
	Ordinal    int                       `json:"ordinal"`
	ItemKey    string                    `json:"itemKey"`
	SubjectKey string                    `json:"subjectKey"`
	Coverage   auditstore.Coverage       `json:"coverage"`
	Result     *auditstore.ExactArtifact `json:"result,omitempty"`
	UpdatedAt  time.Time                 `json:"updatedAt"`
}

type auditCoveragePageResponse struct {
	Items []auditCoverageResponse `json:"items"`
	Page  pageInfoResponse        `json:"page"`
}

type auditReportResponse struct {
	Status          auditservice.ReportStatus `json:"status"`
	MachineArtifact *auditstore.ExactArtifact `json:"machineArtifact,omitempty"`
	SummaryArtifact *auditstore.ExactArtifact `json:"summaryArtifact,omitempty"`
	Machine         json.RawMessage           `json:"machine,omitempty"`
	Summary         string                    `json:"summary,omitempty"`
}

type auditStartResponse struct {
	Audit auditResponse       `json:"audit"`
	Round auditRoundResponse  `json:"round"`
	Items []auditItemResponse `json:"items"`
}

func (h *handler) listAuditProfiles(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1].Ref
		next, cursorErr := h.encodePageCursor("audit-profiles", last.Name+"@"+last.Version)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	writeJSON(w, http.StatusOK, auditProfilePageResponse{Items: items, Page: page})
}

func (h *handler) getAuditProfile(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
		return
	}
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
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
	page := pageInfoResponse{}
	if len(audits) > limit {
		audits = audits[:limit]
		last := audits[len(audits)-1]
		next, cursorErr := h.encodePageCursor(cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.AuditID)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
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
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	if err := requireEmptyBody(w, r); err != nil {
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
	digest := auditStartRequestDigest(auditID, revision)
	started, err := h.dependencies.Audits.Start(r.Context(), auditservice.StartParams{
		OwnerID: principalUserID(r.Context()), AuditID: auditID, ExpectedRevision: revision,
		IdempotencyKey: key, RequestDigest: digest,
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
	writeJSON(w, http.StatusOK, auditStartResponse{
		Audit: audit, Round: auditRoundReadModel(started.Round), Items: items,
	})
}

func (h *handler) listAuditItems(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
	response := make([]auditItemResponse, len(items))
	for index := range items {
		response[index] = auditItemReadModel(items[index])
	}
	writeJSON(w, http.StatusOK, auditItemPageResponse{Items: response, Page: page})
}

func (h *handler) listAuditCoverage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
	page := pageInfoResponse{}
	if len(coverage) > limit {
		coverage = coverage[:limit]
		last := coverage[len(coverage)-1]
		next, cursorErr := h.encodePageCursor(cursorKind, strconv.Itoa(last.Ordinal))
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore, page.NextCursor = true, &next
	}
	items := make([]auditCoverageResponse, len(coverage))
	for index, row := range coverage {
		items[index] = auditCoverageResponse{
			RoundID: row.RoundID, ItemID: row.ItemID, Ordinal: row.Ordinal,
			ItemKey: row.ItemKey, SubjectKey: row.SubjectKey, Coverage: row.Coverage,
			Result: row.Result, UpdatedAt: row.UpdatedAt,
		}
	}
	writeJSON(w, http.StatusOK, auditCoveragePageResponse{Items: items, Page: page})
}

func (h *handler) getAuditReport(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if h.dependencies.Audits == nil {
		h.handleError(w, fmt.Errorf("Audit service is not configured"))
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
		Status: projection.Status, MachineArtifact: projection.MachineArtifact,
		SummaryArtifact: projection.SummaryArtifact, Machine: projection.Machine,
		Summary: projection.Summary,
	})
}

func auditProfileReadModel(source auditservice.ProfileProjection, detail bool) auditProfileResponse {
	profile := source.Profile
	result := auditProfileResponse{
		Ref: profile.Ref, Mode: profile.Mode,
		Standards: append([]config.AuditStandardRef{}, profile.Standards...),
		Inputs:    profile.Inputs, Inventory: profile.Inventory,
		Execution: profile.Execution, Interaction: profile.Interaction,
		ServerCompatible:        source.Compatibility.ServerCompatible,
		RequiresInputValidation: source.Compatibility.RequiresInputValidation,
		CompatibilityReasons:    append([]auditservice.CompatibilityReason{}, source.Compatibility.Reasons...),
	}
	if detail {
		result.Workflows = make(map[string]auditProfileWorkflowResponse, len(profile.Workflows))
		for role, binding := range profile.Workflows {
			result.Workflows[role] = auditProfileWorkflowResponse{
				Workflow: binding.Workflow.Ref, Inputs: binding.Inputs,
				Parameters: binding.Parameters, Outputs: binding.Outputs,
			}
		}
	}
	return result
}

func auditReadModel(source auditstore.Audit) (auditResponse, error) {
	selection, err := auditservice.DecodeDraftSelection(source.InputSelection)
	if err != nil {
		return auditResponse{}, err
	}
	result := auditResponse{
		AuditID: source.AuditID, ProjectID: source.ProjectID,
		Profile: auditProfileIdentityResponse{
			Name: source.Profile.Name, Version: source.Profile.Version, Digest: source.Profile.Digest,
		},
		Inputs: selection.Inputs, Scope: selection.Scope,
		RuntimeLabels: append([]string{}, selection.RuntimeLabels...),
		State:         source.State, Revision: source.Revision, CurrentRoundID: source.CurrentRoundID,
		DispatchState: source.Dispatch, HoldState: source.Hold, DeadlineAt: source.DeadlineAt,
		Limits: source.Limits, ReservedRunCount: source.ReservedRunCount,
		SubmittedRunCount: source.SubmittedRunCount, OutstandingRunCount: source.OutstandingRunCount,
		RetainedEvidenceBytes: source.RetainedEvidenceBytes, EventSequence: source.EventSequence,
		CreatedAt: source.CreatedAt, UpdatedAt: source.UpdatedAt,
		StartedAt: source.StartedAt, FinishedAt: source.FinishedAt,
	}
	if source.StopReason != nil {
		result.StopReason = &auditStopReasonResponse{
			Code: source.StopReason.Code, Message: source.StopReason.Message,
		}
	}
	if len(source.BaselineSnapshot) != 0 {
		baseline, decodeErr := auditservice.DecodeBaseline(source.BaselineSnapshot)
		if decodeErr != nil {
			return auditResponse{}, decodeErr
		}
		skills := make([]auditSkillResponse, 0, len(baseline.Skills))
		for _, skill := range baseline.Skills {
			if skill.Source == nil {
				return auditResponse{}, fmt.Errorf("stored Audit baseline contains an unresolved Skill")
			}
			skills = append(skills, auditSkillResponse{
				Name: skill.Name, Source: *skill.Source,
				SourceDigest: skill.SourceDigest, SourceSize: skill.SourceSize,
			})
		}
		result.Baseline = &auditBaselineResponse{
			Inputs: baseline.Inputs, Scope: baseline.Scope,
			RuntimeLabels: baseline.RuntimeLabels,
			RuntimeConfig: auditRuntimeSnapshotResponse{
				Default: baseline.RuntimeConfig.Default, Labels: baseline.RuntimeConfig.Labels,
			},
			Skills: skills, ProjectHTTPTarget: baseline.ProjectHTTPTarget,
			Inventory: auditBaselineInventoryResponse{
				SourceContentDigest:      baseline.Inventory.SourceContentDigest,
				CanonicalInventoryDigest: baseline.Inventory.CanonicalInventoryDigest,
				Gaps:                     append([]string{}, baseline.Inventory.Gaps...),
				Worklist:                 baseline.Inventory.Worklist,
			},
		}
	}
	return result, nil
}

func auditRoundReadModel(source auditstore.Round) auditRoundResponse {
	return auditRoundResponse{
		RoundID: source.RoundID, Ordinal: source.Ordinal, Manifest: source.Manifest,
		State: source.State, ExpectedItemCount: source.ExpectedItemCount, Revision: source.Revision,
		CreatedAt: source.CreatedAt, UpdatedAt: source.UpdatedAt,
	}
}

func auditItemReadModel(source auditstore.Item) auditItemResponse {
	return auditItemResponse{
		ItemID: source.ItemID, RoundID: source.RoundID, ItemKey: source.ItemKey,
		Ordinal: source.Ordinal, Kind: source.Kind, SubjectKey: source.SubjectKey,
		Task: source.Task, WorkflowRole: source.WorkflowRole, State: source.State,
		FinalDisposition: source.FinalDisposition, AcceptedResult: source.AcceptedResult,
		LastExecutionItemID: source.LastExecutionItemID,
		CreatedAt:           source.CreatedAt, UpdatedAt: source.UpdatedAt,
	}
}

func writeAuditJSON(w http.ResponseWriter, status int, response auditResponse) {
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(response.Revision, 10)))
	writeJSON(w, status, response)
}

func createAuditRequestDigest(projectID string, request createAuditRequest) (string, []string, error) {
	labels, err := runtimeconfig.NormalizeRunLabels(request.RuntimeLabels)
	if err != nil {
		return "", nil, err
	}
	canonical := struct {
		ProjectID     string                           `json:"projectId"`
		Profile       auditProfileSelectorRequest      `json:"profile"`
		Inputs        map[string]contracts.ArtifactRef `json:"inputs"`
		RuntimeLabels []string                         `json:"runtimeLabels"`
		Scope         auditservice.Scope               `json:"scope"`
	}{projectID, request.Profile, request.Inputs, labels, request.Scope}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", nil, err
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), labels, nil
}

func auditStartRequestDigest(auditID string, revision uint64) string {
	encoded, _ := json.Marshal(struct {
		AuditID  string `json:"auditId"`
		Revision uint64 `json:"revision"`
	}{auditID, revision})
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func requireEmptyBody(w http.ResponseWriter, r *http.Request) error {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1))
	if err != nil || len(data) != 0 {
		return fmt.Errorf("%w: request body must be empty", errInvalidRequest)
	}
	return nil
}

func pointerString(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func pointerAuditState(value *auditstore.AuditState) string {
	if value == nil {
		return ""
	}
	return string(*value)
}

func pointerItemState(value *auditstore.ItemState) string {
	if value == nil {
		return ""
	}
	return string(*value)
}
