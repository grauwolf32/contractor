package public

import (
	"context"
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
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

func (h *credentialHandlers) listCredentials(w http.ResponseWriter, r *http.Request) error {
	_, limit, encodedCursor, err := pageQuery(r.URL.RawQuery)
	if err != nil {
		return err
	}
	cursor, err := h.cursors.decodePageCursor(encodedCursor, "credentials", 1)
	if err != nil {
		return err
	}
	after := ""
	if len(cursor) != 0 {
		after = cursor[0]
	}
	records, err := h.ManagedCredentials.ListCredentials(r.Context(), after, pagelimit.FetchRows(limit))
	if err != nil {
		return err
	}
	records, page, err := paginate(h.cursors, records, limit, "credentials", func(last credentials.Record) []string {
		return []string{last.CredentialID}
	})
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, credentialPageResponse{Items: records, Page: page})
	return nil
}

func (h *credentialHandlers) getCredential(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	credentialID := r.PathValue("credentialId")
	if err := (llmgateway.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		return fmt.Errorf("%w: invalid credential ID", errInvalidRequest)
	}
	record, err := h.ManagedCredentials.GetCredential(r.Context(), credentialID)
	if err != nil {
		return err
	}
	writeJSON(w, http.StatusOK, record)
	return nil
}

func (h *credentialHandlers) createCredential(w http.ResponseWriter, r *http.Request) error {
	if err := requireJSONMediaType(r); err != nil {
		return err
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	var request createCredentialRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	label := ""
	if request.Label != nil {
		if *request.Label == "" {
			return fmt.Errorf("%w: credential label must not be empty", errInvalidRequest)
		}
		label = *request.Label
	}
	result, err := h.ManagedCredentials.Create(r.Context(), credentials.CreateRequest{
		CredentialID: request.CredentialID, LLMGateway: request.LLMGateway,
		Label: label, GatewayPolicy: request.GatewayPolicy,
		IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		return err
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, result.Credential.CredentialID,
		); err != nil {
			return err
		}
	}
	if result.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	writeJSON(w, http.StatusCreated, result.Credential)
	return nil
}

func (h *credentialHandlers) deleteCredential(w http.ResponseWriter, r *http.Request) error {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	credentialID := r.PathValue("credentialId")
	if err := (llmgateway.LLMCredentialRef{CredentialID: credentialID}).Validate(); err != nil {
		return fmt.Errorf("%w: invalid credential ID", errInvalidRequest)
	}
	idempotencyKey, err := requireIdempotencyKey(r)
	if err != nil {
		return err
	}
	result, err := h.ManagedCredentials.Delete(r.Context(), credentials.DeleteRequest{
		CredentialID: credentialID, IdempotencyKey: idempotencyKey, ActorID: principalUserID(r.Context()),
	})
	if err != nil {
		return err
	}
	if !result.Replayed {
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsCredential, credentialID,
		); err != nil {
			return err
		}
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}
