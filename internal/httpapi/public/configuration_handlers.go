package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

func (h *catalogHandlers) listConfigurations(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	kind, err := config.ParseConfigurationKind(r.PathValue("kind"))
	if err != nil {
		handleError(w, err)
		return
	}
	values, limit, encodedCursor, err := pageQuery(r.URL.RawQuery, "q", "name")
	if err != nil {
		handleError(w, err)
		return
	}
	_, namePresent := values["name"]
	query, err := parseCatalogQuery(values.Get("q"), values.Get("name"), namePresent)
	if err != nil {
		handleError(w, err)
		return
	}
	// Resources and their fingerprint come from one snapshot, which encoded
	// them once (CAT-R67).
	discovery, err := h.Config.ConfigurationDiscovery(kind)
	if err != nil {
		handleError(w, err)
		return
	}
	resources := discovery.Resources
	cursorKind := catalogPageCursorKind("configurations:"+string(kind), query, discovery.SourceFingerprint)
	cursor, err := h.cursors.decodePageCursor(encodedCursor, cursorKind, 1)
	if err != nil {
		handleError(w, err)
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
		if len(items) == pagelimit.FetchRows(limit) {
			break
		}
	}
	items, page, err := paginate(h.cursors, items, limit, cursorKind, func(last config.ConfigurationResource) []string {
		return []string{last.Ref.Name + "@" + last.Ref.Version}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, configurationPageResponse{Items: items, Page: page})
}

func (h *catalogHandlers) getConfiguration(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	kind, err := config.ParseConfigurationKind(r.PathValue("kind"))
	if err != nil {
		handleError(w, err)
		return
	}
	selector := r.PathValue("name") + "@" + r.PathValue("version")
	if _, err := resolved.ParseSelector(selector); err != nil {
		handleError(w, fmt.Errorf("%w: invalid configuration selector", errInvalidRequest))
		return
	}
	resource, err := h.Config.Configuration(kind, selector)
	if err != nil {
		handleError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(resource.Ref.Digest))
	writeJSON(w, http.StatusOK, resource)
}

func (h *catalogHandlers) publishConfiguration(w http.ResponseWriter, r *http.Request) {
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, fmt.Errorf("%w: Content-Type must be application/json", errInvalidRequest))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	kind, err := config.ParseConfigurationKind(r.PathValue("kind"))
	if err != nil {
		handleError(w, err)
		return
	}
	var request publishConfigurationRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.ConfigurationPublisher.Publish(r.Context(), config.PublicationRequest{
		Kind: kind, Name: request.Name, Version: request.Version,
		ModelPolicy: request.ModelPolicy, LLMGateway: request.LLMGateway,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsConfiguration, "",
		); err != nil {
			handleError(w, err)
			return
		}
	}
	w.Header().Set("ETag", strconv.Quote(result.Resource.Ref.Digest))
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Resource)
}
