package public

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

// maximumGitImportBodyBytes bounds the JSON body of a Git import request
// (LIM-GIT-5).
const maximumGitImportBodyBytes int64 = 64 << 10

type GitImportService interface {
	Timeout() time.Duration
	DoImport(context.Context, gitimport.ImportRequest, func(gitimport.ImportResult)) error
}

func (h *artifactHandlers) importGitArtifact(w http.ResponseWriter, r *http.Request) error {
	ctx, cancel := context.WithTimeout(r.Context(), h.GitImports.Timeout())
	defer cancel()
	r = r.WithContext(ctx)
	deadline, _ := ctx.Deadline()
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(deadline)
	_ = controller.SetWriteDeadline(deadline)
	defer func() { _ = controller.SetReadDeadline(time.Time{}); _ = controller.SetWriteDeadline(time.Time{}) }()
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if err := validateArtifactRouteNames(r); err != nil {
		return err
	}
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	expected, err := artifactWritePrecondition(r)
	if err != nil {
		return err
	}
	var body struct {
		RepositoryURL string          `json:"repositoryUrl"`
		Mode          json.RawMessage `json:"mode"`
		Ref           json.RawMessage `json:"ref"`
	}
	if err := decodeJSONBounded(w, r, &body, maximumGitImportBodyBytes); err != nil {
		return err
	}
	mode := gitimport.ModeSnapshot
	if len(body.Mode) != 0 {
		mode = ""
		if json.Unmarshal(body.Mode, &mode) != nil || (mode != gitimport.ModeSnapshot && mode != gitimport.ModeAnalysis) {
			return errInvalidRequest
		}
	}
	ref := ""
	if len(body.Ref) != 0 {
		if json.Unmarshal(body.Ref, &ref) != nil || ref == "" {
			return gitimport.ErrRef
		}
	}
	err = h.GitImports.DoImport(r.Context(), gitimport.ImportRequest{
		Mode: mode, OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"), Target: artifacts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}, ExpectedRevision: expected, RepositoryURL: body.RepositoryURL, Ref: ref,
	}, func(result gitimport.ImportResult) {
		// Import admission is still held here; bound this write from now, not
		// from the start of the 120-second import.
		artifacttransfer.BoundWrite(w, 0)
		status := http.StatusCreated
		if expected != nil {
			status = http.StatusOK
		}
		w.Header().Set("ETag", httpx.QuotedETag(result.Artifact.Revision))
		writeJSON(w, status, result)
	})
	if err != nil {
		return err
	}
	return nil
}
