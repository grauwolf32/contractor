package public

import (
	"encoding/json"
	"net/http"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/evaldomain"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// handleError answers err with the public error envelope of its Kind: the
// Kind that the first classified error of its chain reports (apierror), the
// retryable 503 storage_transaction_conflict for a definite database
// transaction conflict, and the retryable 500 internal_error for any other
// failure. The answer never carries the error's internal text.
func handleError(w http.ResponseWriter, err error) {
	requestid.AnnotateFailure(w, "", diagnosticCause(err))
	classified, ok := apierror.Of(err)
	switch {
	case ok:
		writeErrorDetails(w, classified.ErrorKind(), apierror.MessageOf(classified), errorDetails(classified))
	case persistencepostgres.IsTransactionConflict(err):
		writeError(w, apierror.StorageTransactionConflict, "")
	default:
		writeError(w, apierror.InternalError, "")
	}
}

// errorDetails returns the safe details of a classified error that has them.
// Their kind is the error's own code, so details and code cannot disagree;
// only the managed Eval details carry the kind eval with a recovery action.
func errorDetails(err apierror.Classified) any {
	kind := err.ErrorKind().Code()
	switch typed := err.(type) {
	case *auditservice.UnsupportedError:
		return &auditUnsupportedDetailsResponse{
			Kind:    kind,
			Reasons: append([]auditservice.CompatibilityReason(nil), typed.Reasons...),
		}
	case *runtimeconfig.LabelInUseError:
		return &runtimeLabelInUseDetailsResponse{
			Kind: kind, RuntimeAgentIDs: append([]string(nil), typed.RuntimeAgentIDs...),
		}
	case *credentials.RuntimeCredentialInUseError:
		return &runtimeCredentialInUseDetailsResponse{
			Kind:          kind,
			BindingLabels: append([]string(nil), typed.Usage.BindingLabels...),
			ProjectIDs:    append([]string(nil), typed.Usage.ProjectIDs...),
			RunIDs:        append([]string(nil), typed.Usage.RunIDs...),
			AuditIDs:      append([]string(nil), typed.Usage.AuditIDs...),
			AllocationIDs: append([]string(nil), typed.Usage.AllocationIDs...),
		}
	case *credentials.CredentialInUseError:
		return &credentialInUseDetailsResponse{
			Kind: kind, RunIDs: append([]string{}, typed.RunIDs...),
			AuditIDs:      append([]string(nil), typed.AuditIDs...),
			BindingLabels: append([]string(nil), typed.BindingLabels...),
		}
	case *rundeletion.NotDeletableError:
		return &runNotDeletableDetailsResponse{Kind: kind, Reason: typed.Reason}
	case *evaldomain.Error:
		// No Eval failure is retryable as sent; its recovery action says what
		// to do instead (FAIL-RUN-1 for eval_idempotency_conflict).
		return map[string]string{"kind": "eval", "recovery": typed.Recovery}
	}
	return nil
}

// writeError answers kind with message, or with the Kind's own message when
// message is empty. A Kind is a registered code, so every error the public
// API writes is a code of the registry with its status and retryability.
func writeError(w http.ResponseWriter, kind apierror.Kind, message string) {
	writeErrorDetails(w, kind, message, nil)
}

func writeErrorDetails(w http.ResponseWriter, kind apierror.Kind, message string, details any) {
	if message == "" {
		message = kind.Message()
	}
	writeJSON(w, kind.Status(), errorResponse{
		Code: kind.Code(), Message: message, Retryable: kind.Retryable(),
		RequestID: requestid.FromResponse(w), Details: details,
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func methodNotAllowed(w http.ResponseWriter, _ *http.Request) {
	writeError(w, apierror.MethodNotAllowed, "")
}

func rejectHead(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodHead {
		return false
	}
	methodNotAllowed(w, r)
	return true
}

func notFound(w http.ResponseWriter, _ *http.Request) {
	writeError(w, apierror.NotFound, "")
}
