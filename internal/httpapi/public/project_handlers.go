package public

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

type ProjectManagement interface {
	Create(context.Context, projectstore.CreateParams) (projectstore.Project, bool, error)
	Get(context.Context, string, string) (projectstore.Project, error)
	List(context.Context, projectstore.ListParams) ([]projectstore.Project, error)
	Update(context.Context, projectstore.UpdateParams) (projectstore.Project, error)
	BeginDeletion(context.Context, projectstore.BeginDeletionParams) (projectstore.Project, bool, error)
}

// ProjectCredentialValidator checks that the principal may attach a Runtime
// credential to a Project HTTP target, under the lock that keeps the
// credential from being deleted while the target is written.
type ProjectCredentialValidator interface {
	ValidateRuntimeCredentialUse(context.Context, credentials.RuntimeCredentialUser, string, ...string) error
	WithCredentialReferences(context.Context, func() error) error
}

// ProjectDependencies serve the Project resource and start Project deletion.
type ProjectDependencies struct {
	Projects                ProjectManagement
	Sitemap                 ProjectSitemap
	RuntimeCredentials      ProjectCredentialValidator
	ProjectDeletionNotifier Waker
	NewID                   func(string) (string, error)
}

type projectHandlers struct {
	ProjectDependencies
	cursors pageCursors
}

func (h *projectHandlers) createProject(w http.ResponseWriter, r *http.Request) error {
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
	var request createProjectRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	params := projectstore.CreateParams{
		OwnerID: principalUserID(r.Context()), Kind: request.Kind,
		Name: request.Name, Description: request.Description,
		IdempotencyKey: key, ActiveChecks: request.ActiveChecks,
	}
	params.RequestDigest, err = projectstore.CreateRequestDigest(params)
	if err != nil {
		return err
	}
	params.ProjectID, err = h.NewID("project_")
	if err != nil {
		return fmt.Errorf("generate Project ID: %w", err)
	}
	project, created, err := h.Projects.Create(r.Context(), params)
	if err != nil {
		return err
	}
	if !created {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeProject(w, http.StatusCreated, project)
	return nil
}

func (h *projectHandlers) listProjects(w http.ResponseWriter, r *http.Request) error {
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "kind")
	if err != nil {
		return err
	}
	var kind *projectstore.Kind
	if values, present := query["kind"]; present {
		candidate := projectstore.Kind(values[0])
		if !candidate.Valid() {
			return errInvalidRequest
		}
		kind = &candidate
	}
	cursorKind := "projects"
	if kind != nil {
		cursorKind += ":" + string(*kind)
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		return err
	}
	params := projectstore.ListParams{
		OwnerID: principalUserID(r.Context()), Kind: kind, Limit: pagelimit.FetchRows(limit),
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			return errInvalidRequest
		}
		params.BeforeCreatedAt = &before
		params.BeforeProjectID = cursor[1]
	}
	projects, err := h.Projects.List(r.Context(), params)
	if err != nil {
		return err
	}
	projects, page, err := paginate(h.cursors, projects, limit, cursorKind, func(last projectstore.Project) []string {
		return []string{last.CreatedAt.UTC().Format(time.RFC3339Nano), last.ProjectID}
	})
	if err != nil {
		return err
	}
	items := make([]projectResponse, 0, len(projects))
	for _, project := range projects {
		items = append(items, projectReadModel(project))
	}
	writeJSON(w, http.StatusOK, projectPageResponse{Items: items, Page: page})
	return nil
}

func (h *projectHandlers) getProject(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	project, err := h.Projects.Get(
		r.Context(), principalUserID(r.Context()), r.PathValue("projectId"),
	)
	if err != nil {
		return err
	}
	writeProject(w, http.StatusOK, project)
	return nil
}

func (h *projectHandlers) updateProject(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	revision, err := requiredRevision(r, "Project update")
	if err != nil {
		return err
	}
	var request updateProjectRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	ownerID, projectID := principalUserID(r.Context()), r.PathValue("projectId")
	current, err := h.Projects.Get(r.Context(), ownerID, projectID)
	if err != nil {
		return err
	}
	name, description := current.Name, current.Description
	httpTarget := current.HTTPTarget.Clone()
	if request.Name != nil {
		name = *request.Name
	}
	if request.Description != nil {
		description = *request.Description
	}
	if request.targetSet {
		httpTarget = nil
		if request.HTTPTarget != nil {
			httpTarget = &runtimesettings.HTTPOriginTargetRef{
				URL: request.HTTPTarget.URL, Credential: request.HTTPTarget.Credential,
			}
		}
	}
	var project projectstore.Project
	update := func() error {
		var updateErr error
		project, updateErr = h.Projects.Update(r.Context(), projectstore.UpdateParams{
			ProjectID: projectID, OwnerID: ownerID, ExpectedRevision: revision,
			Name: name, Description: description, HTTPTarget: httpTarget, ActiveChecks: request.ActiveChecks,
		})
		return updateErr
	}
	if request.targetSet && httpTarget != nil && httpTarget.Credential != nil {
		err = h.RuntimeCredentials.WithCredentialReferences(r.Context(), func() error {
			if validateErr := h.RuntimeCredentials.ValidateRuntimeCredentialUse(
				r.Context(), runtimeCredentialUser(r.Context()),
				httpTarget.Credential.CredentialID, string(httpTarget.Credential.Kind),
			); validateErr != nil {
				return validateErr
			}
			return update()
		})
	} else {
		err = update()
	}
	if err != nil {
		return err
	}
	writeProject(w, http.StatusOK, project)
	return nil
}

func (h *projectHandlers) deleteProject(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil ||
		r.ContentLength > 0 || len(r.TransferEncoding) != 0 {
		return errInvalidRequest
	}
	revision, err := requiredRevision(r, "Project deletion")
	if err != nil {
		return err
	}
	project, _, err := h.Projects.BeginDeletion(r.Context(), projectstore.BeginDeletionParams{
		ProjectID: r.PathValue("projectId"), OwnerID: principalUserID(r.Context()),
		ExpectedRevision: revision,
	})
	if err != nil {
		return err
	}
	h.ProjectDeletionNotifier.Wake()
	writeProject(w, http.StatusAccepted, project)
	return nil
}

func writeProject(w http.ResponseWriter, status int, project projectstore.Project) {
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(project.Revision, 10)))
	writeJSON(w, status, projectReadModel(project))
}

func projectReadModel(project projectstore.Project) projectResponse {
	result := projectResponse{
		ProjectID: project.ProjectID, Kind: project.Kind, Name: project.Name,
		Description: project.Description, Lifecycle: project.Lifecycle, ActiveChecks: project.ActiveChecks,
		Revision:   strconv.FormatUint(project.Revision, 10),
		HTTPTarget: project.HTTPTarget.Clone(),
		CreatedAt:  project.CreatedAt, UpdatedAt: project.UpdatedAt,
	}
	if project.Deletion != nil {
		result.Deletion = &projectDeletionResponse{
			Phase: project.Deletion.Phase, RequestedAt: project.Deletion.RequestedAt,
		}
	}
	return result
}
