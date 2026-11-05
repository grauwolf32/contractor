package pgstore

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func validateID(field, value string) error {
	if !contracts.ValidResourceID(value) {
		return invalidf("%s is invalid", field)
	}
	return nil
}

func validateText(field, value string, maximum int, required bool) error {
	if !utf8.ValidString(value) || len([]byte(value)) > maximum || required && strings.TrimSpace(value) == "" {
		return invalidf("%s is invalid", field)
	}
	return nil
}

func validateDigest(field, value string) error {
	if !contentdigest.Valid(value) {
		return invalidf("%s is invalid", field)
	}
	return nil
}

func validateIdempotency(key, digest string) error {
	if !contracts.ValidIdempotencyKey(key) {
		return invalidf("idempotency key is invalid")
	}
	return validateDigest("request digest", digest)
}

func validateJSONObject(field string, value json.RawMessage, maximum int) error {
	if len(value) == 0 || len(value) > maximum || !json.Valid(value) {
		return invalidf("%s is invalid", field)
	}
	decoder := json.NewDecoder(bytes.NewReader(value))
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return invalidf("%s is invalid", field)
	}
	if _, ok := decoded.(map[string]any); !ok {
		return invalidf("%s must be a JSON object", field)
	}
	return nil
}

func validateLimits(value auditstore.Limits) error {
	if value.MaxRounds < 1 || value.MaxRounds > 32 || value.BatchSize < 1 || value.BatchSize > 64 ||
		value.MaxItemsPerRound < 1 || value.MaxItemsPerRound > 10_000 ||
		value.MaxItemsTotal < 1 || value.MaxItemsTotal > 100_000 ||
		value.MaxSubmittedRuns < 1 || value.MaxSubmittedRuns > 1_000_000 ||
		value.MaxItemRunAttempts < 1 || value.MaxItemRunAttempts > 10 ||
		value.MaxEvidenceBytes < 1 || value.MaxEvidenceBytes > 1<<30 {
		return invalidf("Audit limits are invalid")
	}
	if value.BatchSize > value.MaxItemsPerRound || value.MaxItemsPerRound > value.MaxItemsTotal ||
		value.MaxSubmittedRuns < (value.MaxItemsTotal+value.BatchSize-1)/value.BatchSize {
		return invalidf("Audit limits are inconsistent")
	}
	return nil
}

func validateCreate(params auditstore.CreateDraftParams) error {
	if err := validateID("auditID", params.AuditID); err != nil {
		return err
	}
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if err := validateID("projectID", params.ProjectID); err != nil {
		return err
	}
	if err := validateText("profile name", params.Profile.Name, 128, true); err != nil {
		return err
	}
	if err := validateText("profile version", params.Profile.Version, 128, true); err != nil {
		return err
	}
	if err := validateDigest("profile digest", params.Profile.Digest); err != nil {
		return err
	}
	if err := validateJSONObject("profile snapshot", params.ProfileSnapshot, 8<<20); err != nil {
		return err
	}
	if err := validateJSONObject("input selection", params.InputSelection, 8<<20); err != nil {
		return err
	}
	if err := validateLimits(params.Limits); err != nil {
		return err
	}
	return validateIdempotency(params.IdempotencyKey, params.RequestDigest)
}

func validateList(params auditstore.ListParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if params.ProjectID != nil {
		if err := validateID("projectID", *params.ProjectID); err != nil {
			return err
		}
	}
	for _, state := range params.States {
		if !state.Valid() {
			return invalidf("Audit state is invalid")
		}
	}
	if (params.ProfileName == nil) != (params.ProfileVersion == nil) {
		return invalidf("Audit profile filter is incomplete")
	}
	if params.ProfileName != nil {
		if err := validateText("profile name", *params.ProfileName, 128, true); err != nil {
			return err
		}
		if err := validateText("profile version", *params.ProfileVersion, 128, true); err != nil {
			return err
		}
	}
	// HTTP pages fetch one extra row to determine whether a cursor is needed.
	if !pagelimit.ValidFetch(params.Limit) {
		return invalidf("page row limit must be between 1 and %d", pagelimit.MaxFetchRows)
	}
	if (params.BeforeCreatedAt == nil) != (params.BeforeAuditID == "") {
		return invalidf("Audit keyset is incomplete")
	}
	if params.BeforeCreatedAt != nil {
		if params.BeforeCreatedAt.IsZero() {
			return invalidf("Audit keyset timestamp is invalid")
		}
		return validateID("beforeAuditID", params.BeforeAuditID)
	}
	return nil
}

func validateMutationReplay(
	ownerID string, operation auditstore.MutationOperation, key, digest string,
) error {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return err
	}
	if operation != auditstore.MutationCreate && operation != auditstore.MutationStart &&
		operation != auditstore.MutationTransition && operation != auditstore.MutationDelete {
		return invalidf("Audit mutation operation is invalid")
	}
	return validateIdempotency(key, digest)
}

func validateDelete(params auditstore.DeleteParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if err := validateID("auditID", params.AuditID); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 {
		return invalidf("Audit delete revision is invalid")
	}
	return validateIdempotency(params.IdempotencyKey, params.RequestDigest)
}

func transitionAllowed(from, to auditstore.AuditState) bool {
	switch from {
	case auditstore.AuditDraft:
		return to == auditstore.AuditActive || to == auditstore.AuditDeleting
	case auditstore.AuditActive:
		return to == auditstore.AuditPaused || to == auditstore.AuditFinalizing || to == auditstore.AuditCancelling
	case auditstore.AuditPaused:
		return to == auditstore.AuditActive || to == auditstore.AuditCancelling
	case auditstore.AuditFinalizing:
		return to == auditstore.AuditCompleted || to == auditstore.AuditFailed || to == auditstore.AuditCancelling
	case auditstore.AuditCancelling:
		return to == auditstore.AuditCancelled || to == auditstore.AuditDeleting
	case auditstore.AuditCompleted, auditstore.AuditCancelled, auditstore.AuditFailed:
		return to == auditstore.AuditDeleting
	default:
		return false
	}
}

func validateTransition(params auditstore.TransitionParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if err := validateID("auditID", params.AuditID); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 || !params.ExpectedState.Valid() || !params.TargetState.Valid() ||
		!transitionAllowed(params.ExpectedState, params.TargetState) {
		return invalidf("Audit transition is invalid")
	}
	if !ownerTransitionAllowed(params.ExpectedState, params.TargetState) {
		return invalidf("Audit transition requires Controller authority")
	}
	if params.Reason != nil {
		if err := validateText("reason code", params.Reason.Code, 128, true); err != nil {
			return err
		}
		if err := validateText("reason message", params.Reason.Message, auditstore.MaxStopReasonMessageBytes, false); err != nil {
			return err
		}
	}
	return validateIdempotency(params.IdempotencyKey, params.RequestDigest)
}

func validateResume(params auditstore.ResumeParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if err := validateID("auditID", params.AuditID); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 ||
		!transitionAllowed(auditstore.AuditPaused, auditstore.AuditActive) {
		return invalidf("Audit resume is invalid")
	}
	return validateIdempotency(params.IdempotencyKey, params.RequestDigest)
}

// ownerTransitionAllowed lists the plain owner transitions. Paused to active
// is deliberately absent: only Resume may reopen dispatch, because it also
// renews the hold and deadline behind the Project gate.
func ownerTransitionAllowed(from, to auditstore.AuditState) bool {
	if to == auditstore.AuditPaused {
		return from == auditstore.AuditActive
	}
	if to == auditstore.AuditCancelling {
		return from == auditstore.AuditActive ||
			from == auditstore.AuditPaused || from == auditstore.AuditFinalizing
	}
	return false
}

// ValidateClaimedTransition is the check TransitionClaimed applies before its
// compare-and-set: it returns ErrInvalid for a claim identity, revision or
// transition the store refuses (transitionAllowed, without draft to active,
// which only the start of preparation performs). Test doubles of
// TransitionClaimed apply it too, so they reject what the store rejects.
func ValidateClaimedTransition(params auditstore.ClaimedTransitionParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 || !params.ExpectedState.Valid() ||
		!params.TargetState.Valid() || !transitionAllowed(params.ExpectedState, params.TargetState) ||
		(params.ExpectedState == auditstore.AuditDraft && params.TargetState == auditstore.AuditActive) {
		return invalidf("claimed Audit transition is invalid")
	}
	if params.Reason != nil {
		if err := validateText("reason code", params.Reason.Code, 128, true); err != nil {
			return err
		}
		return validateText("reason message", params.Reason.Message, auditstore.MaxStopReasonMessageBytes, false)
	}
	return nil
}

func validateExactArtifact(field string, value auditstore.ExactArtifact, requireMetadata bool) error {
	if err := value.Ref.ValidateExact(); err != nil {
		return invalidf("%s ref is invalid", field)
	}
	if !utf8.ValidString(value.Ref.Namespace) || !utf8.ValidString(value.Ref.Name) ||
		value.Ref.Revision == nil || !utf8.ValidString(*value.Ref.Revision) {
		return invalidf("%s ref is invalid", field)
	}
	encodedRef, err := json.Marshal(value.Ref)
	if err != nil || len(encodedRef) > auditstore.MaxArtifactRefBytes {
		return invalidf("%s ref is too large", field)
	}
	if err := validateDigest(field+" digest", value.Digest); err != nil {
		return err
	}
	if requireMetadata || value.MediaType != "" || value.SizeBytes != 0 {
		metadata := contracts.ArtifactReadResult{
			APIVersion: contracts.APIVersion,
			Artifact:   value.Ref,
			MediaType:  value.MediaType,
			Size:       value.SizeBytes,
		}
		if err := metadata.Validate(); err != nil {
			return invalidf("%s metadata is invalid", field)
		}
	}
	return nil
}

func validateClaimIdentity(claim auditstore.ControllerClaim) error {
	if err := validateID("auditID", claim.AuditID); err != nil {
		return err
	}
	if err := validateText("claim holder", claim.HolderID, 256, true); err != nil {
		return err
	}
	if claim.Epoch == 0 || claim.Epoch > math.MaxInt64 {
		return invalidf("claim epoch is invalid")
	}
	return nil
}

func validateClaimParams(params auditstore.ClaimParams) error {
	if err := validateText("claim holder", params.HolderID, 256, true); err != nil {
		return err
	}
	if !persistencepostgres.ValidControllerLease(params.Lease) || params.Limit < 1 || params.Limit > auditstore.MaxClaimBatch {
		return invalidf("claim bounds are invalid")
	}
	return nil
}

func validateBindRun(params auditstore.BindRunParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if err := validateID("executionID", params.ExecutionID); err != nil {
		return err
	}
	return validateID("runID", params.RunID)
}

func validateObserveTerminal(params auditstore.ObserveTerminalParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if err := validateID("executionID", params.ExecutionID); err != nil {
		return err
	}
	if err := validateID("runID", params.RunID); err != nil {
		return err
	}
	if err := validateText("Run event generation", params.Generation, 256, true); err != nil {
		return err
	}
	if params.Sequence == 0 || params.Sequence > math.MaxInt64 {
		return invalidf("Run event sequence is invalid")
	}
	return nil
}

func validateSubmissionFailure(params auditstore.ObserveSubmissionFailureParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	return validateID("executionID", params.ExecutionID)
}

func validateArtifactLinks(links []auditstore.ArtifactLink) (int64, error) {
	seenLinks := make(map[string]struct{}, len(links))
	var retainedBytes int64
	retainedArtifacts := make(map[string]struct{}, len(links))
	for _, link := range links {
		if link.Scope != "project" && link.Scope != "audit" {
			return 0, invalidf("artifact scope is invalid")
		}
		if err := validateText("artifact logical key", link.LogicalKey, auditstore.MaxArtifactLogicalKeyBytes, true); err != nil {
			return 0, err
		}
		if _, duplicate := seenLinks[link.LogicalKey]; duplicate {
			return 0, invalidf("artifact logical key is duplicated")
		}
		seenLinks[link.LogicalKey] = struct{}{}
		if err := validateExactArtifact("retained artifact", link.Artifact, true); err != nil {
			return 0, err
		}
		if err := validateJSONObject("artifact source provenance", link.SourceProvenance, 1<<20); err != nil {
			return 0, err
		}
		if err := validateText("artifact display ref", link.DisplayRef, 1024, false); err != nil {
			return 0, err
		}
		artifactKey := retainedArtifactKey(link)
		if _, counted := retainedArtifacts[artifactKey]; !counted {
			if link.Artifact.SizeBytes > 1<<30-retainedBytes {
				return 0, invalidf("retained artifact bytes overflow")
			}
			retainedArtifacts[artifactKey] = struct{}{}
			retainedBytes += link.Artifact.SizeBytes
		}
	}
	encoded, err := json.Marshal(links)
	if err != nil || len(encoded) > auditstore.MaxRetainedRefsBytes {
		return 0, invalidf("retained artifact refs are too large")
	}
	return retainedBytes, nil
}
