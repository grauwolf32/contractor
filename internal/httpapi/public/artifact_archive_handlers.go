package public

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifactpreview"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

type artifactArchiveResponse struct {
	Artifact contracts.ArtifactRef   `json:"artifact"`
	Entries  []artifactpreview.Entry `json:"entries"`
}

type artifactArchiveFileResponse struct {
	Artifact contracts.ArtifactRef `json:"artifact"`
	Path     string                `json:"path"`
	Size     int                   `json:"size"`
	Text     string                `json:"text"`
}

type archiveStoreResolver func(http.ResponseWriter, *http.Request) (artifacts.ScopedStore, bool)

func (h *handler) registerArchiveRoutes(mux *http.ServeMux, prefix string, resolve archiveStoreResolver) {
	for _, file := range []bool{false, true} {
		path := prefix + "/archive"
		if file {
			path += "/file"
		}
		mux.HandleFunc("GET "+path, func(w http.ResponseWriter, r *http.Request) {
			h.getArtifactArchive(w, r, resolve, file)
		})
		mux.HandleFunc(path, h.methodNotAllowed)
	}
}

func (h *handler) userArchiveStore(w http.ResponseWriter, r *http.Request) (artifacts.ScopedStore, bool) {
	if err := validateUserArtifactReadRoute(r); err != nil {
		h.handleError(w, err)
		return artifacts.ScopedStore{}, false
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
	}
	return store, err == nil
}

func (h *handler) projectArchiveStore(w http.ResponseWriter, r *http.Request) (artifacts.ScopedStore, bool) {
	store, _, err := h.ownedProjectArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return artifacts.ScopedStore{}, false
	}
	return store, !h.rejectAuditManagedProjectArtifact(w, r)
}

func (h *handler) runArchiveStore(w http.ResponseWriter, r *http.Request) (artifacts.ScopedStore, bool) {
	store, _, err := h.ownedRunArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return artifacts.ScopedStore{}, false
	}
	return store, !h.rejectRunSystemArtifactRoute(w, r)
}

func (h *handler) getArtifactArchive(w http.ResponseWriter, r *http.Request, resolve archiveStoreResolver, file bool) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if h.rejectHead(w, r) {
		return
	}
	allowed := []string{"revision"}
	if file {
		allowed = append(allowed, "path")
	}
	query, err := exactQuery(r.URL.RawQuery, allowed...)
	if err != nil {
		h.handleError(w, err)
		return
	}
	revision := query.Get("revision")
	if err := validatePublicRevision(revision); err != nil {
		h.handleError(w, err)
		return
	}
	if err := validateArtifactRouteNames(r); err != nil {
		h.handleError(w, err)
		return
	}
	if file && !artifactpreview.ValidPath(query.Get("path")) {
		h.handleError(w, errInvalidRequest)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	ctx, release, err := artifacts.AcquireTransfer(ctx)
	if err != nil {
		h.handleError(w, err)
		return
	}
	defer release()
	r = r.WithContext(ctx)
	store, ok := resolve(w, r)
	if !ok {
		return
	}
	result, err := store.Read(ctx, contracts.ArtifactRef{
		Namespace: r.PathValue("namespace"), Name: r.PathValue("name"), Revision: &revision,
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !artifactpreview.SupportsMediaType(result.Payload.MediaType) {
		h.writeError(w, http.StatusUnsupportedMediaType, "archive_media_type", "Archive preview requires a ZIP or Skill package", false)
		return
	}
	archive, err := artifactpreview.Open(ctx, result.Payload.Data)
	if err != nil {
		h.handleArchiveError(w, err)
		return
	}
	if file {
		text, err := archive.Text(ctx, query.Get("path"))
		if err != nil {
			h.handleArchiveError(w, err)
			return
		}
		w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
		writeJSON(w, http.StatusOK, artifactArchiveFileResponse{
			Artifact: result.Ref, Path: query.Get("path"), Size: len(text), Text: text,
		})
		return
	}
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	writeJSON(w, http.StatusOK, artifactArchiveResponse{Artifact: result.Ref, Entries: archive.Entries})
}

func (h *handler) handleArchiveError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, artifactpreview.ErrInvalid):
		h.writeError(w, http.StatusUnprocessableEntity, "archive_invalid", "Archive is invalid, unsupported or contains unsafe entries. Download the original to inspect it separately.", false)
	case errors.Is(err, artifactpreview.ErrLimit):
		h.writeError(w, http.StatusUnprocessableEntity, "archive_preview_limit", "Archive preview limit exceeded. Text files must fit 256 KiB; the original archive remains downloadable.", false)
	case errors.Is(err, artifactpreview.ErrNotText):
		h.writeError(w, http.StatusUnprocessableEntity, "archive_file_not_text", "This file is not supported UTF-8 text. Download the original archive to view it separately.", false)
	case errors.Is(err, artifactpreview.ErrNotFound):
		h.handleError(w, artifacts.ErrArtifactNotFound)
	default:
		h.handleError(w, err)
	}
}
