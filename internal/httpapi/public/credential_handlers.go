package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
)

func (h *handler) listCredentials(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		h.handleError(w, err)
		return
	}
	cursor, err := h.decodePageCursor(encodedCursor, "credentials", 1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	records, err := h.dependencies.ManagedCredentials.ListCredentials(r.Context(), after, limit+1)
	if err != nil {
		h.handleError(w, err)
		return
	}
	page := pageInfoResponse{}
	if len(records) > limit {
		records = records[:limit]
		next, cursorErr := h.encodePageCursor("credentials", records[len(records)-1].CredentialID)
		if cursorErr != nil {
			h.handleError(w, cursorErr)
			return
		}
		page.HasMore = true
		page.NextCursor = &next
	}
	writeJSON(w, http.StatusOK, credentialPageResponse{Items: records, Page: page})
}

func (h *handler) getCredential(w http.ResponseWriter, r *http.Request) {
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	credentialID := r.PathValue("credentialId")
	if err := (contracts.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		h.handleError(w, fmt.Errorf("%w: invalid credential ID", errInvalidRequest))
		return
	}
	record, err := h.dependencies.ManagedCredentials.GetCredential(r.Context(), credentialID)
	if err != nil {
		h.handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, record)
}

func (h *handler) createCredential(w http.ResponseWriter, r *http.Request) {
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, fmt.Errorf("%w: Content-Type must be application/json", errInvalidRequest))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	var request createCredentialRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	label := ""
	if request.Label != nil {
		if *request.Label == "" {
			h.handleError(w, fmt.Errorf("%w: credential label must not be empty", errInvalidRequest))
			return
		}
		label = *request.Label
	}
	result, err := h.dependencies.ManagedCredentials.Create(r.Context(), credentials.CreateRequest{
		CredentialID: request.CredentialID, LLMGateway: request.LLMGateway,
		Label: label, GatewayPolicy: request.GatewayPolicy,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		h.handleError(w, err)
		return
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Credential)
}

func (h *handler) deleteCredential(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	credentialID := r.PathValue("credentialId")
	if err := (contracts.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		h.handleError(w, fmt.Errorf("%w: invalid credential ID", errInvalidRequest))
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	if _, err := h.dependencies.ManagedCredentials.Delete(r.Context(), credentials.DeleteRequest{
		CredentialID: credentialID, IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	}); err != nil {
		h.handleError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
