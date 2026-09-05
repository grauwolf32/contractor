package public

import (
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func (h *handler) listArtifacts(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactBindings(w, r, store, "user-artifacts", true)
}

func (h *handler) getArtifactMetadata(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.getArtifactMetadataFromStore(w, r, store)
}

func (h *handler) listArtifactVersions(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactVersionsFromStore(w, r, store, "user-artifact-versions")
}

func (h *handler) listArtifactLineage(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactLineageFromStore(w, r, store, "user-artifact-lineage")
}

func (h *handler) listRunArtifacts(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, runID, err := h.ownedRunArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactBindings(w, r, store, "run-artifacts:"+runID, false)
}

func (h *handler) getRunArtifact(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		h.methodNotAllowed(w, r)
		return
	}
	store, _, err := h.ownedRunArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Read(r.Context(), ref)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeArtifactBytes(w, result)
}

func (h *handler) getRunArtifactMetadata(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, _, err := h.ownedRunArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.getArtifactMetadataFromStore(w, r, store)
}

func (h *handler) listRunArtifactVersions(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, runID, err := h.ownedRunArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactVersionsFromStore(w, r, store, "run-artifact-versions:"+runID)
}

func (h *handler) listRunArtifactLineage(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	store, runID, err := h.ownedRunArtifactStore(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.listArtifactLineageFromStore(w, r, store, "run-artifact-lineage:"+runID)
}

func (h *handler) ownedRunArtifactStore(r *http.Request) (artifacts.ScopedStore, string, error) {
	run, err := h.ownedRun(r)
	if err != nil {
		return artifacts.ScopedStore{}, "", err
	}
	store, err := h.dependencies.Artifacts.Run(run.RunID)
	return store, run.RunID, err
}

func (h *handler) listArtifactBindings(
	w http.ResponseWriter,
	r *http.Request,
	store artifacts.ScopedStore,
	cursorPrefix string,
	allowNamespaceExclusion bool,
) {
	allowedQuery := []string{"namespace"}
	if allowNamespaceExclusion {
		allowedQuery = append(allowedQuery, "excludeNamespace")
	}
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, allowedQuery...)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var namespace *string
	if values, present := query["namespace"]; present {
		value := values[0]
		if err := validatePublicArtifactName(value); err != nil {
			h.handleError(w, err)
			return
		}
		namespace = &value
	}
	cursorKind := cursorPrefix
	if namespace != nil {
		cursorKind += ":" + *namespace
	}
	var excludeNamespace *string
	if values, present := query["excludeNamespace"]; present {
		value := values[0]
		if err := validatePublicArtifactName(value); err != nil {
			h.handleError(w, err)
			return
		}
		excludeNamespace = &value
		cursorKind += ":exclude:" + value
	}
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	pageQuery := artifacts.BindingPageQuery{
		Namespace: namespace, ExcludeNamespace: excludeNamespace, Limit: limit + 1,
	}
	if len(cursor) != 0 {
		pageQuery.AfterNamespace, pageQuery.AfterName = cursor[0], cursor[1]
	}
	items, err := store.ListMetadata(r.Context(), pageQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1].Ref
		next, cursorErr := h.encodePageCursor(cursorKind, last.Namespace, last.Name)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, artifactPageResponse{Items: items, Page: page})
}

func (h *handler) getArtifactMetadataFromStore(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore,
) {
	ref, err := artifactRouteRef(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	metadata, err := store.Metadata(r.Context(), ref)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, metadata)
}

func (h *handler) listArtifactVersionsFromStore(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, cursorPrefix string,
) {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if err := validateArtifactRouteNames(r); err != nil {
		h.handleError(w, err)
		return
	}
	cursorKind := cursorPrefix + ":" + ref.Namespace + ":" + ref.Name
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 2)
	if err != nil {
		h.handleError(w, err)
		return
	}
	pageQuery := artifacts.VersionPageQuery{Limit: limit + 1}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[0])
		if parseErr != nil {
			h.handleError(w, fmt.Errorf("%w: invalid Artifact cursor", errInvalidRequest))
			return
		}
		pageQuery.BeforeCreatedAt = &before
		pageQuery.BeforeRevision = cursor[1]
	}
	items, err := store.ListVersions(r.Context(), ref, pageQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind, last.CreatedAt.UTC().Format(time.RFC3339Nano), *last.Ref.Revision,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, artifactPageResponse{Items: items, Page: page})
}

func (h *handler) listArtifactLineageFromStore(
	w http.ResponseWriter, r *http.Request, store artifacts.ScopedStore, cursorPrefix string,
) {
	query, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "revision")
	if err != nil {
		h.handleError(w, err)
		return
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if err := validateArtifactRouteNames(r); err != nil {
		h.handleError(w, err)
		return
	}
	if values, present := query["revision"]; present {
		if err := validatePublicRevision(values[0]); err != nil {
			h.handleError(w, err)
			return
		}
		ref.Revision = &values[0]
	}
	cursorKind := cursorPrefix + ":" + ref.Namespace + ":" + ref.Name
	if ref.Revision != nil {
		cursorKind += ":" + *ref.Revision
	}
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 5)
	if err != nil {
		h.handleError(w, err)
		return
	}
	selectedRevision := ""
	if len(cursor) != 0 {
		selectedRevision = cursor[0]
		if ref.Revision != nil && *ref.Revision != selectedRevision {
			h.handleError(w, fmt.Errorf("%w: Artifact cursor selects another revision", errInvalidRequest))
			return
		}
		ref.Revision = &selectedRevision
	} else {
		metadata, metadataErr := store.Metadata(r.Context(), ref)
		if metadataErr != nil {
			h.handleError(w, metadataErr)
			return
		}
		selectedRevision = *metadata.Ref.Revision
		ref = metadata.Ref
	}
	pageQuery := artifacts.LineagePageQuery{Limit: limit + 1}
	if len(cursor) != 0 {
		before, parseErr := time.Parse(time.RFC3339Nano, cursor[1])
		if parseErr != nil {
			h.handleError(w, fmt.Errorf("%w: invalid Artifact cursor", errInvalidRequest))
			return
		}
		pageQuery.BeforeCreatedAt = &before
		pageQuery.BeforeTargetRevision = cursor[2]
		pageQuery.BeforeSourceRevision = cursor[3]
		pageQuery.BeforeKind = cursor[4]
	}
	items, err := store.ListLineage(r.Context(), ref, pageQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1]
		next, cursorErr := h.encodePageCursor(
			cursorKind, selectedRevision, last.CreatedAt.UTC().Format(time.RFC3339Nano),
			*last.Target.Revision, *last.Source.Revision, last.Kind,
		)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, artifactLineagePageResponse{Items: items, Page: page})
}

func artifactRouteRef(r *http.Request) (contracts.ArtifactRef, error) {
	query, err := exactQuery(r.URL.RawQuery, "revision")
	if err != nil {
		return contracts.ArtifactRef{}, err
	}
	ref := contracts.ArtifactRef{Namespace: r.PathValue("namespace"), Name: r.PathValue("name")}
	if err := validateArtifactRouteNames(r); err != nil {
		return contracts.ArtifactRef{}, err
	}
	if values, present := query["revision"]; present {
		if err := validatePublicRevision(values[0]); err != nil {
			return contracts.ArtifactRef{}, err
		}
		ref.Revision = &values[0]
	}
	return ref, nil
}

func validateArtifactRouteNames(r *http.Request) error {
	if err := validatePublicArtifactName(r.PathValue("namespace")); err != nil {
		return err
	}
	return validatePublicArtifactName(r.PathValue("name"))
}

func writeArtifactBytes(w http.ResponseWriter, result artifacts.ReadResult) {
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}

func (h *handler) putArtifact(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	if err := validateArtifactRouteNames(r); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	expectedRevision, err := artifactWritePrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	payload, err := readArtifactBody(w, r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Write(r.Context(), contracts.ArtifactRef{
		Namespace: r.PathValue("namespace"),
		Name:      r.PathValue("name"),
	}, artifacts.Payload{MediaType: mediaType, Data: payload}, expectedRevision)
	if err != nil {
		h.handleError(w, err)
		return
	}

	status := http.StatusCreated
	if expectedRevision != nil {
		status = http.StatusOK
	}
	w.Header().Set("ETag", quotedETag(result.Ref.Revision))
	writeJSON(w, status, artifactWriteResponse{
		Artifact:  result.Ref,
		MediaType: result.MediaType,
		Size:      result.Size,
	})
}

func (h *handler) getArtifact(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		h.methodNotAllowed(w, r)
		return
	}
	ref, err := artifactRouteRef(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	store, err := h.dependencies.Artifacts.User(principalUserID(r.Context()))
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := store.Read(r.Context(), ref)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeArtifactBytes(w, result)
}
