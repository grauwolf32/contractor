package public

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/gitimport"
)

type GitImportService interface {
	DoImport(context.Context, gitimport.ImportRequest, func(gitimport.ImportResult)) error
}

func (h *handler) importGitArtifact(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), gitimport.Deadline)
	defer cancel()
	r = r.WithContext(ctx)
	deadline, _ := ctx.Deadline()
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(deadline)
	_ = controller.SetWriteDeadline(deadline)
	defer func() { _ = controller.SetReadDeadline(time.Time{}); _ = controller.SetWriteDeadline(time.Time{}) }()
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	if err := validateArtifactRouteNames(r); err != nil {
		h.handleError(w, err)
		return
	}
	mt, err := requestMediaType(r)
	if err != nil || mt != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	expected, err := artifactWritePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var body struct {
		RepositoryURL string          `json:"repositoryUrl"`
		Ref           json.RawMessage `json:"ref"`
	}
	if err := decodeJSONBounded(w, r, &body, 64<<10); err != nil {
		h.handleError(w, err)
		return
	}
	ref := ""
	if len(body.Ref) != 0 {
		if json.Unmarshal(body.Ref, &ref) != nil || ref == "" {
			h.gitImportError(w, gitimport.ErrRef)
			return
		}
	}
	if h.dependencies.GitImports == nil {
		h.gitImportError(w, gitimport.ErrConfiguration)
		return
	}
	err = h.dependencies.GitImports.DoImport(r.Context(), gitimport.ImportRequest{
		OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"), Target: artifacts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}, ExpectedRevision: expected, RepositoryURL: body.RepositoryURL, Ref: ref,
	}, func(result gitimport.ImportResult) {
		status := http.StatusCreated
		if expected != nil {
			status = http.StatusOK
		}
		w.Header().Set("ETag", quotedETag(result.Artifact.Revision))
		writeJSON(w, status, result)
	})
	if err != nil {
		h.gitImportError(w, err)
	}
}
func (h *handler) gitImportError(w http.ResponseWriter, err error) {
	for _, item := range []struct {
		err           error
		status        int
		code, message string
	}{
		{gitimport.ErrURL, 400, "git_url_invalid", "Git repository URL is invalid"},
		{gitimport.ErrRef, 422, "git_ref_unavailable", "Git branch or tag is missing or ambiguous"},
		{gitimport.ErrDestination, 422, "git_destination_refused", "Git remote destination is not allowed"},
		{gitimport.ErrTrust, 422, "git_host_untrusted", "Git SSH host trust is unavailable or verification failed"},
		{gitimport.ErrRemote, 422, "git_remote_unavailable", "Git repository is unavailable or access was refused"},
		{gitimport.ErrContent, 422, "git_content_unsupported", "Git snapshot contains unsupported or invalid content"},
		{gitimport.ErrBudget, 413, "git_import_limit", "Git import exceeds a resource limit"},
		{gitimport.ErrCapacity, 503, "git_import_capacity", "Another Git import is active"},
		{gitimport.ErrConfiguration, 503, "git_import_unavailable", "Git import is unavailable"},
		{context.DeadlineExceeded, 504, "git_import_timeout", "Git import timed out; inspect artifact metadata before retrying"},
		{context.Canceled, 408, "git_import_cancelled", "Git import was interrupted; inspect artifact metadata before retrying"},
	} {
		if errors.Is(err, item.err) {
			h.writeError(w, item.status, item.code, item.message, false)
			return
		}
	}
	h.gitKeyError(w, err)
}
