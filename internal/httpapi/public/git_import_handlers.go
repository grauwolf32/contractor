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

type GitImportService interface {
	DoImport(context.Context, gitimport.ImportRequest, func(gitimport.ImportResult)) error
}

func (h *artifactHandlers) importGitArtifact(w http.ResponseWriter, r *http.Request) {
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
		handleError(w, err)
		return
	}
	if err := validateArtifactRouteNames(r); err != nil {
		handleError(w, err)
		return
	}
	mt, err := requestMediaType(r)
	if err != nil || mt != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	expected, err := artifactWritePrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var body struct {
		RepositoryURL string          `json:"repositoryUrl"`
		Ref           json.RawMessage `json:"ref"`
	}
	if err := decodeJSONBounded(w, r, &body, 64<<10); err != nil {
		handleError(w, err)
		return
	}
	ref := ""
	if len(body.Ref) != 0 {
		if json.Unmarshal(body.Ref, &ref) != nil || ref == "" {
			handleError(w, gitimport.ErrRef)
			return
		}
	}
	err = h.GitImports.DoImport(r.Context(), gitimport.ImportRequest{
		OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId"), Target: artifacts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}, ExpectedRevision: expected, RepositoryURL: body.RepositoryURL, Ref: ref,
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
		handleError(w, err)
	}
}
