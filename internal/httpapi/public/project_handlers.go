package public

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

func (h *handler) createProject(w http.ResponseWriter, r *http.Request) {
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
	var request createProjectRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	key, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	digest, err := projectRequestDigest(request)
	if err != nil {
		h.handleError(w, err)
		return
	}
	projectID, err := h.dependencies.NewID("project_")
	if err != nil {
		h.handleError(w, fmt.Errorf("generate Project ID: %w", err))
		return
	}
	project, created, err := h.dependencies.Projects.Create(r.Context(), projectstore.CreateParams{
		ProjectID: projectID, OwnerID: principalUserID(r.Context()), Kind: request.Kind,
		Name: request.Name, Description: request.Description,
		IdempotencyKey: key, RequestDigest: digest,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !created {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeProject(w, http.StatusCreated, project)
}

func (h *handler) listProjects(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "kind")
	if err != nil {
		h.handleError(w, err)
		return
	}
	var kind *projectstore.Kind
	if values, present := query["kind"]; present {
		candidate := projectstore.Kind(values[0])
		if !candidate.Valid() {
			h.handleError(w, errInvalidRequest)
			return
		}
		kind = &candidate
	}
	cursorKind := "projects"
	if kind != nil {
		cursorKind += ":" + string(*kind)
	}
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	params := projectstore.ListParams{
		OwnerID: principalUserID(r.Context()), Kind: kind, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, errInvalidRequest)
			return
		}
		params.BeforeCreatedAt = &before
		params.BeforeProjectID = cursor[1]
	}
	projects, err := h.dependencies.Projects.List(r.Context(), params)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(projects) > limit {
		projects = projects[:limit]
		last := projects[len(projects)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), last.ProjectID,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	items := make([]projectResponse, 0, len(projects))
	for _, project := range projects {
		items = append(items, projectReadModel(project))
	}
	writeJSON(w, http.StatusOK, projectPageResponse{Items: items, Page: page})
}

func (h *handler) getProject(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	project, err := h.dependencies.Projects.Get(
		r.Context(), principalUserID(r.Context()), r.PathValue("projectId"),
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeProject(w, http.StatusOK, project)
}

func (h *handler) updateProject(w http.ResponseWriter, r *http.Request) {
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
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		h.handleError(w, fmt.Errorf("%w: Project update requires one If-Match", errInvalidRequest))
		return
	}
	revision, err := parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request updateProjectRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	ownerID, projectID := principalUserID(r.Context()), r.PathValue("projectId")
	current, err := h.dependencies.Projects.Get(r.Context(), ownerID, projectID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	name, description := current.Name, current.Description
	if request.Name != nil {
		name = *request.Name
	}
	if request.Description != nil {
		description = *request.Description
	}
	project, err := h.dependencies.Projects.Update(r.Context(), projectstore.UpdateParams{
		ProjectID: projectID, OwnerID: ownerID, ExpectedRevision: revision,
		Name: name, Description: description,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeProject(w, http.StatusOK, project)
}

func projectRequestDigest(request createProjectRequest) (string, error) {
	encoded, err := json.Marshal(struct {
		Kind        projectstore.Kind `json:"kind"`
		Name        string            `json:"name"`
		Description string            `json:"description"`
	}{request.Kind, request.Name, request.Description})
	if err != nil {
		return "", fmt.Errorf("encode Project request: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func writeProject(w http.ResponseWriter, status int, project projectstore.Project) {
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(project.Revision, 10)))
	writeJSON(w, status, projectReadModel(project))
}

func projectReadModel(project projectstore.Project) projectResponse {
	return projectResponse{
		ProjectID: project.ProjectID, Kind: project.Kind, Name: project.Name,
		Description: project.Description, Revision: strconv.FormatUint(project.Revision, 10),
		CreatedAt: project.CreatedAt, UpdatedAt: project.UpdatedAt,
	}
}
