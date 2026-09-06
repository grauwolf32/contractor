package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

func (h *handler) listConfigurations(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	kind, err := config.ParseConfigurationKind(r.PathValue("kind"))
	if err != nil {
		h.handleError(w, err)
		return
	}
	values, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "q", "name")
	if err != nil {
		h.handleError(w, err)
		return
	}
	_, namePresent := values["name"]
	query, err := parseCatalogQuery(values.Get("q"), values.Get("name"), namePresent)
	if err != nil {
		h.handleError(w, err)
		return
	}
	resources, err := h.dependencies.Config.Configurations(kind)
	if err != nil {
		h.handleError(w, err)
		return
	}
	sourceFingerprint, err := catalogSourceFingerprint(resources)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursorKind := catalogPageCursorKind("configurations:"+string(kind), query, sourceFingerprint)
	cursor, err := h.decodePageCursor(encodedCursor, cursorKind, 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	items := make([]config.ConfigurationResource, 0, min(limit, len(resources)))
	for _, resource := range resources {
		if !catalogMatches(
			query, resource.Ref.Name, resource.Ref.Version, configurationDescription(resource),
		) {
			continue
		}
		selector := resource.Ref.Name + "@" + resource.Ref.Version
		if selector <= after {
			continue
		}
		items = append(items, resource)
		if len(items) == limit+1 {
			break
		}
	}
	page := pageInfoResponse{}
	if len(items) > limit {
		items = items[:limit]
		last := items[len(items)-1].Ref
		next, cursorErr := h.encodePageCursor(cursorKind, last.Name+"@"+last.Version)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, configurationPageResponse{Items: items, Page: page})
}

func (h *handler) getConfiguration(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	kind, err := config.ParseConfigurationKind(r.PathValue("kind"))
	if err != nil {
		h.handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := config.ParseSelector(selector); err != nil {
		h.handleError(w, fmt.Errorf("%w: invalid configuration selector", errInvalidRequest))
		return
	}
	resource, err := h.dependencies.Config.Configuration(kind, selector)
	if err != nil {
		h.handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(resource.Ref.Digest))
	writeJSON(w, http.StatusOK, resource)
}

func (h *handler) publishConfiguration(w http.ResponseWriter, r *http.Request) {
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, fmt.Errorf("%w: Content-Type must be application/json", errInvalidRequest))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	kind, err := config.ParseConfigurationKind(r.PathValue("kind"))
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request publishConfigurationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	result, err := h.dependencies.ConfigurationPublisher.Publish(r.Context(), config.PublicationRequest{
		Kind: kind, Name: request.Name, Version: request.Version,
		ModelPolicy: request.ModelPolicy, LLMGateway: request.LLMGateway,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, "",
		); err != nil {
			h.handleError(w, err)
			return
		}
	}
	w.Header().Set("ETag", strconv.Quote(result.Resource.Ref.Digest))
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Resource)
}
