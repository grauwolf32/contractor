package public

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
)

type AuditManagement interface {
	WriteHypothesis(context.Context, auditservice.HypothesisMutationParams) (auditstore.HypothesisAcceptance, bool, error)
	Hypotheses(context.Context, auditstore.ListHypothesesParams) (auditstore.HypothesisPage, error)
	Hypothesis(context.Context, string, string, string, uint64) (auditstore.Hypothesis, error)
	CreateTasks(context.Context, auditservice.TaskCreationParams) (auditservice.TaskCreationResult, error)
	Tasks(context.Context, auditstore.ListTasksParams) (auditstore.TaskPage, error)
	Task(context.Context, string, string, string) (auditstore.Task, error)
	PublishMaterial(context.Context, string, string, string, auditservice.MaterialPublicationRequest, *string) (auditservice.MaterialPublication, error)
	ListProjectFindings(context.Context, auditservice.ProjectFindingListParams) (auditservice.ProjectFindingPage, error)
	GetProjectFinding(context.Context, string, string, string) (auditstore.ProjectFinding, error)
	GetAuditProjectFinding(context.Context, string, string, string) (auditstore.ProjectFinding, error)
	EditProjectFinding(context.Context, auditservice.ProjectFindingEditParams) (auditservice.ProjectFindingEditResult, error)
	FindingCandidates(context.Context, auditservice.FindingCandidatesParams) (auditservice.FindingCandidatesPage, error)
	CompareFindings(context.Context, auditservice.FindingComparisonParams) (auditservice.FindingComparisonPage, error)
	ListEventsPage(context.Context, auditstore.EventListParams) (auditservice.EventPage, error)
	ListOwnerFindings(context.Context, auditstore.OwnerFindingListParams) (auditstore.FindingWindow, error)
	GetWorkspace(context.Context, string, string) (auditstore.WorkspaceSummary, error)
	ListFindingsPage(context.Context, auditstore.FindingListParams) (auditservice.FindingPage, error)
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
	ReadArtifact(context.Context, string, string, string, string) (artifacts.ReadResult, error)
	List(context.Context, auditstore.ListParams) ([]auditstore.Audit, error)
	GetReport(context.Context, string, string) (auditservice.ReportProjection, error)
}

// AuditDependencies serve AuditProfiles, AuditStandards, Audits, their
// items, coverage, events, findings, reviews and reports, and the owner-wide
// Audit, finding and review lists.
type AuditDependencies struct {
	Transfers artifacttransfer.Admission
	Audits    AuditManagement
	Projects  ProjectReader
	NewID     func(string) (string, error)
}

type auditHandlers struct {
	AuditDependencies
	cursors pageCursors
}

func (h *auditHandlers) listAuditProfiles(w http.ResponseWriter, r *http.Request) error {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "audit-profiles", 1)
	if err != nil {
		return err
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
		return err
	}
	writeJSON(w, http.StatusOK, auditProfilePageResponse{Items: items, Page: page})
	return nil
}

func (h *auditHandlers) getAuditProfile(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	selector := auditservice.ProfileSelector{Name: r.PathValue("name"), Version: r.PathValue("version")}
	if _, err := resolved.ParseSelector(selector.Name + "@" + selector.Version); err != nil {
		return errInvalidRequest
	}
	profile, err := h.Audits.Profile(selector)
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(profile.Profile.Ref.Digest))
	writeJSON(w, http.StatusOK, auditProfileReadModel(profile, true))
	return nil
}

func (h *auditHandlers) createAudit(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	var request createAuditRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	params := auditservice.CreateDraftParams{
		Labels:  runlabels.RunMetadataLabels(request.Labels).Clone(),
		OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"),
		OperationsPrincipal: runtimeCredentialUser(r.Context()).Operations,
		Profile:             auditservice.ProfileSelector{Name: request.Profile.Name, Version: request.Profile.Version},
		Inputs:              request.Inputs, RuntimeLabels: request.RuntimeLabels, Scope: request.Scope,
		IdempotencyKey: key,
	}
	params.RequestDigest, err = auditservice.CreateDraftRequestDigest(params)
	if err != nil {
		return err
	}
	params.AuditID, err = h.NewID("audit_")
	if err != nil {
		return fmt.Errorf("generate Audit ID: %w", err)
	}
	audit, created, err := h.Audits.CreateDraft(r.Context(), params)
	if err != nil {
		return err
	}
	if !created {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	response, err := auditReadModel(audit)
	if err != nil {
		return err
	}
	writeAuditJSON(w, http.StatusCreated, response)
	return nil
}

func (h *auditHandlers) listProjectAudits(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	projectID := r.PathValue("projectId")
	if projectID != "" {
		if _, err := h.Projects.Get(r.Context(), principalUserID(r.Context()), projectID); err != nil {
			return err
		}
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "state", "profile")
	if err != nil {
		return err
	}
	// The owner-wide list accepts a comma-separated state set; a Project list
	// filters by one state.
	var states []auditstore.AuditState
	if values, ok := query["state"]; ok {
		if projectID == "" {
			states, err = parseAuditStates(values[0])
			if err != nil {
				return err
			}
		} else {
			candidate := auditstore.AuditState(values[0])
			if !candidate.Valid() {
				return errInvalidRequest
			}
			states = []auditstore.AuditState{candidate}
		}
	}
	var profileName, profileVersion *string
	profileSelector := ""
	if values, ok := query["profile"]; ok {
		selector, parseErr := resolved.ParseSelector(values[0])
		if parseErr != nil {
			return errInvalidRequest
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
		return err
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
			return errInvalidRequest
		}
		params.BeforeCreatedAt, params.BeforeAuditID = &before, cursor[1]
	}
	audits, err := h.Audits.List(r.Context(), params)
	if err != nil {
		return err
	}
	audits, page, err := paginate(h.cursors, audits, limit, cursorKind, func(last auditstore.Audit) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.AuditID}
	})
	if err != nil {
		return err
	}
	items := make([]auditResponse, 0, len(audits))
	for _, audit := range audits {
		item, modelErr := auditReadModel(audit)
		if modelErr != nil {
			return modelErr
		}

		items = append(items, item)
	}
	writeJSON(w, http.StatusOK, auditPageResponse{Items: items, Page: page})
	return nil
}

func (h *auditHandlers) getAudit(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	audit, err := h.Audits.Get(r.Context(), principalUserID(r.Context()), r.PathValue("auditId"))
	if err != nil {
		return err
	}
	response, err := auditReadModel(audit)
	if err != nil {
		return err
	}
	writeAuditJSON(w, http.StatusOK, response)
	return nil
}

func (h *auditHandlers) startAudit(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireOptionalJSONMediaType(r); err != nil {
		return err
	}
	key, revision, err := auditMutationHeaders(r)
	if err != nil {
		return err
	}
	seconds, err := readAuditTimeLimit(w, r)
	if err != nil {
		return err
	}
	auditID := r.PathValue("auditId")
	digest, err := auditservice.LifecycleRequestDigest("start", auditID, revision, seconds)
	if err != nil {
		return err
	}
	credentialUser := runtimeCredentialUser(r.Context())
	started, err := h.Audits.Start(r.Context(), auditservice.StartParams{
		OwnerID: credentialUser.UserID, OperationsPrincipal: credentialUser.Operations,
		AuditID: auditID, ExpectedRevision: revision,
		IdempotencyKey: key, RequestDigest: digest, DeadlineSeconds: seconds,
	})
	if err != nil {
		return err
	}
	if started.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	audit, err := auditReadModel(started.Audit)
	if err != nil {
		return err
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(audit.Revision, 10)))
	writeJSON(w, http.StatusOK, auditStartResponse{Audit: audit})
	return nil
}

func (h *auditHandlers) pauseAudit(w http.ResponseWriter, r *http.Request) error {
	return h.mutateAudit(w, r, "pause", http.StatusOK)
}

func (h *auditHandlers) resumeAudit(w http.ResponseWriter, r *http.Request) error {
	return h.mutateAudit(w, r, "resume", http.StatusOK)
}

func (h *auditHandlers) cancelAudit(w http.ResponseWriter, r *http.Request) error {
	return h.mutateAudit(w, r, "cancel", http.StatusAccepted)
}

func (h *auditHandlers) deleteAudit(w http.ResponseWriter, r *http.Request) error {
	return h.mutateAudit(w, r, "delete", http.StatusAccepted)
}

func (h *auditHandlers) mutateAudit(w http.ResponseWriter, r *http.Request, action string, status int) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if action == "resume" {
		if err := requireOptionalJSONMediaType(r); err != nil {
			return err
		}
	}
	key, revision, err := auditMutationHeaders(r)
	if err != nil {
		return err
	}
	var seconds *int
	if action == "resume" {
		seconds, err = readAuditTimeLimit(w, r)
	} else {
		err = requireEmptyBody(w, r)
	}
	if err != nil {
		return err
	}
	params := auditservice.MutationParams{
		OwnerID: principalUserID(r.Context()), AuditID: r.PathValue("auditId"),
		ExpectedRevision: revision, IdempotencyKey: key, DeadlineSeconds: seconds,
	}
	params.RequestDigest, err = auditservice.LifecycleRequestDigest(action, params.AuditID, revision, seconds)
	if err != nil {
		return err
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
		return err
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	response, err := auditReadModel(result.Audit)
	if err != nil {
		return err
	}
	writeAuditJSON(w, status, response)
	return nil
}

func (h *auditHandlers) getAuditReport(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	projection, err := h.Audits.GetReport(
		r.Context(), principalUserID(r.Context()), r.PathValue("auditId"),
	)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, auditReportResponse{
		Status: projection.Status, MachineArtifact: projection.MachineArtifact,
		SummaryArtifact: projection.SummaryArtifact, Machine: projection.Machine,
		Summary: projection.Summary,
	})
	return nil
}
