package public

import (
	"context"
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

type ManagedCredentialLifecycle interface {
	ListCredentials(context.Context, string, int) ([]credentials.Record, error)
	GetCredential(context.Context, string) (credentials.Record, error)
	Create(context.Context, credentials.CreateRequest) (credentials.CreateResult, error)
	Delete(context.Context, credentials.DeleteRequest) (credentials.DeleteResult, error)
}

// CredentialDependencies serve the managed LLM credentials.
type CredentialDependencies struct {
	ManagedCredentials    ManagedCredentialLifecycle
	OperationsInvalidator OperationsInvalidator
}

type credentialHandlers struct {
	CredentialDependencies
	cursors pageCursors
}

func (h *credentialHandlers) listCredentials(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		handleError(w, err)
		return
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "credentials", 1)
	if err != nil {
		handleError(w, err)
		return
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	records, err := h.ManagedCredentials.ListCredentials(r.Context(), after, pagelimit.FetchRows(limit))
	if err != nil {
		handleError(w, err)
		return
	}
	records, page, err := paginate(h.cursors, records, limit, "credentials", func(last credentials.Record) []string {
		return []string{last.CredentialID}
	})
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, credentialPageResponse{Items: records, Page: page})
}

func (h *credentialHandlers) getCredential(w http.ResponseWriter, r *http.Request) {
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	credentialID := r.PathValue("credentialId")
	if err := (llmgateway.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		handleError(w, fmt.Errorf("%w: invalid credential ID", errInvalidRequest))
		return
	}
	record, err := h.ManagedCredentials.GetCredential(r.Context(), credentialID)
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, record)
}

func (h *credentialHandlers) createCredential(w http.ResponseWriter, r *http.Request) {
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, fmt.Errorf("%w: Content-Type must be application/json", errInvalidRequest))
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	var request createCredentialRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	label := ""
	if request.Label != nil {
		if *request.Label == "" {
			handleError(w, fmt.Errorf("%w: credential label must not be empty", errInvalidRequest))
			return
		}
		label = *request.Label
	}
	result, err := h.ManagedCredentials.Create(r.Context(), credentials.CreateRequest{
		CredentialID: request.CredentialID, LLMGateway: request.LLMGateway,
		Label: label, GatewayPolicy: request.GatewayPolicy,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, result.Credential.CredentialID,
		); err != nil {
			handleError(w, err)
			return
		}
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Credential)
}

func (h *credentialHandlers) deleteCredential(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	credentialID := r.PathValue("credentialId")
	if err := (llmgateway.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		handleError(w, fmt.Errorf("%w: invalid credential ID", errInvalidRequest))
		return
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		handleError(w, err)
		return
	}
	result, err := h.ManagedCredentials.Delete(r.Context(), credentials.DeleteRequest{
		CredentialID: credentialID, IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		handleError(w, err)
		return
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, credentialID,
		); err != nil {
			handleError(w, err)
			return
		}
	}
	w.WriteHeader(http.StatusNoContent)
}
