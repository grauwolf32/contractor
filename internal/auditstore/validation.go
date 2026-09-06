package auditstore

import (
	"bytes"
	"encoding/json"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var (
	resourceIDPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)
	idempotencyKeyPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	digestPattern         = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	eventKindPattern      = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,127}$`)
)

func validateID(field, value string) error {
	if !resourceIDPattern.MatchString(value) {
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
	if !digestPattern.MatchString(value) {
		return invalidf("%s is invalid", field)
	}
	return nil
}

func validateIdempotency(key, digest string) error {
	if !idempotencyKeyPattern.MatchString(key) {
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

func validateLimits(value Limits) error {
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

func validateCreate(params CreateDraftParams) error {
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

func validateList(params ListParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if params.ProjectID != nil {
		if err := validateID("projectID", *params.ProjectID); err != nil {
			return err
		}
	}
	if params.State != nil && !params.State.Valid() {
		return invalidf("Audit state is invalid")
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
	if params.Limit < 1 || params.Limit > MaxPageSize {
		return invalidf("page limit must be between 1 and %d", MaxPageSize)
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

func validateListItems(params ListItemsParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if err := validateID("auditID", params.AuditID); err != nil {
		return err
	}
	if params.RoundID != nil {
		if err := validateID("roundID", *params.RoundID); err != nil {
			return err
		}
	}
	if params.State != nil && !params.State.Valid() {
		return invalidf("Audit item state is invalid")
	}
	if params.SubjectKey != nil {
		if err := validateText("subjectKey", *params.SubjectKey, 512, true); err != nil {
			return err
		}
	}
	if params.Limit < 1 || params.Limit > MaxPageSize {
		return invalidf("item page limit must be between 1 and %d", MaxPageSize)
	}
	keysetPresent := params.AfterRoundOrdinal != nil || params.AfterItemOrdinal != nil || params.AfterItemID != ""
	if keysetPresent && (params.AfterRoundOrdinal == nil || params.AfterItemOrdinal == nil || params.AfterItemID == "") {
		return invalidf("item keyset is incomplete")
	}
	if keysetPresent {
		if *params.AfterRoundOrdinal < 1 || *params.AfterItemOrdinal < 0 {
			return invalidf("item keyset ordinal is invalid")
		}
		if err := validateID("afterItemID", params.AfterItemID); err != nil {
			return err
		}
	}
	return nil
}

func validateMutationReplay(
	ownerID string, operation MutationOperation, key, digest string,
) error {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return err
	}
	if operation != MutationCreate && operation != MutationStart &&
		operation != MutationTransition && operation != MutationDelete {
		return invalidf("Audit mutation operation is invalid")
	}
	return validateIdempotency(key, digest)
}

func validateDelete(params DeleteParams) error {
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

func transitionAllowed(from, to AuditState) bool {
	switch from {
	case AuditDraft:
		return to == AuditActive || to == AuditDeleting
	case AuditActive:
		return to == AuditWaitingReview || to == AuditPaused || to == AuditFinalizing || to == AuditCancelling
	case AuditWaitingReview:
		return to == AuditActive || to == AuditPaused || to == AuditFinalizing || to == AuditCancelling
	case AuditPaused:
		return to == AuditActive || to == AuditCancelling
	case AuditFinalizing:
		return to == AuditWaitingReview || to == AuditCompleted || to == AuditFailed || to == AuditCancelling
	case AuditCancelling:
		return to == AuditCancelled || to == AuditDeleting
	case AuditCompleted, AuditCancelled, AuditFailed:
		return to == AuditDeleting
	default:
		return false
	}
}

func validateTransition(params TransitionParams) error {
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
		if err := validateText("reason message", params.Reason.Message, 4096, false); err != nil {
			return err
		}
	}
	return validateIdempotency(params.IdempotencyKey, params.RequestDigest)
}

func ownerTransitionAllowed(from, to AuditState) bool {
	if to == AuditPaused {
		return from == AuditActive || from == AuditWaitingReview
	}
	if from == AuditPaused && to == AuditActive {
		return true
	}
	if to == AuditCancelling {
		return from == AuditActive || from == AuditWaitingReview ||
			from == AuditPaused || from == AuditFinalizing
	}
	if to == AuditDeleting {
		return from == AuditDraft || from == AuditCompleted || from == AuditCancelled ||
			from == AuditFailed || from == AuditCancelling
	}
	return false
}

func validateClaimedTransition(params ClaimedTransitionParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 || !params.ExpectedState.Valid() ||
		!params.TargetState.Valid() || !transitionAllowed(params.ExpectedState, params.TargetState) ||
		(params.ExpectedState == AuditDraft && params.TargetState == AuditActive) {
		return invalidf("claimed Audit transition is invalid")
	}
	if params.Reason != nil {
		if err := validateText("reason code", params.Reason.Code, 128, true); err != nil {
			return err
		}
		return validateText("reason message", params.Reason.Message, 4096, false)
	}
	return nil
}

func roundTransitionAllowed(from, to RoundState) bool {
	switch from {
	case RoundProposed:
		return to == RoundAccepted
	case RoundAccepted:
		return to == RoundExecuting || to == RoundClosed
	case RoundExecuting:
		return to == RoundAssessing || to == RoundClosed
	case RoundAssessing:
		return to == RoundClosed
	default:
		return false
	}
}

func validateRoundTransition(params RoundTransitionParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if err := validateID("roundID", params.RoundID); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 || !params.ExpectedState.Valid() ||
		!params.TargetState.Valid() || !roundTransitionAllowed(params.ExpectedState, params.TargetState) {
		return invalidf("Audit round transition is invalid")
	}
	return nil
}

func validateExactArtifact(field string, value ExactArtifact, requireMetadata bool) error {
	if err := value.Ref.ValidateExact(); err != nil {
		return invalidf("%s ref is invalid", field)
	}
	if !utf8.ValidString(value.Ref.Namespace) || !utf8.ValidString(value.Ref.Name) ||
		value.Ref.Revision == nil || !utf8.ValidString(*value.Ref.Revision) {
		return invalidf("%s ref is invalid", field)
	}
	encodedRef, err := json.Marshal(value.Ref)
	if err != nil || len(encodedRef) > MaxArtifactRefBytes {
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

func validateCoverage(value Coverage) error {
	if !value.Status.Valid() || len(value.Requested) > 4096 || len(value.Completed) > 4096 || len(value.Gaps) > 4096 {
		return invalidf("coverage is invalid")
	}
	if err := validateText("coverage rationale", value.Rationale, 4096, false); err != nil {
		return err
	}
	for _, values := range [][]string{value.Requested, value.Completed, value.Gaps} {
		seen := make(map[string]struct{}, len(values))
		for _, entry := range values {
			if err := validateText("coverage value", entry, 512, true); err != nil {
				return err
			}
			if _, duplicate := seen[entry]; duplicate {
				return invalidf("coverage contains a duplicate value")
			}
			seen[entry] = struct{}{}
		}
		encoded, err := json.Marshal(values)
		if err != nil || len(encoded) > MaxCoverageArrayBytes {
			return invalidf("coverage values are too large")
		}
	}
	return nil
}

func validateItemOrigin(value ItemOrigin, itemKey string, allowIncomplete bool) error {
	if value.Schema != ItemOriginSchema || value.EntryKey != itemKey {
		return invalidf("item origin identity is invalid")
	}
	if value.ProvenanceIncomplete {
		if !allowIncomplete {
			return invalidf("new item origin cannot be incomplete")
		}
		return nil
	}
	if value.SourceRef == nil || value.SourceRef.ValidateExact() != nil {
		return invalidf("item origin source ref is invalid")
	}
	if err := validateDigest("item origin source digest", value.SourceContentDigest); err != nil {
		return err
	}
	if err := validateDigest("item origin inventory digest", value.CanonicalInventoryDigest); err != nil {
		return err
	}
	if err := validateText("item origin source media type", value.SourceMediaType, 256, true); err != nil {
		return err
	}
	if err := validateText("item origin entry version", value.EntryVersion, 256, false); err != nil {
		return err
	}
	encoded, err := json.Marshal(value)
	if err != nil || len(encoded) > 16<<10 {
		return invalidf("item origin is too large")
	}
	return nil
}

func validateMaterialize(params MaterializeRoundParams) error {
	if err := validateText("ownerID", params.OwnerID, 256, true); err != nil {
		return err
	}
	if err := validateID("auditID", params.AuditID); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > math.MaxInt64 || params.RoundOrdinal != 1 {
		return invalidf("round precondition is invalid")
	}
	if err := validateID("roundID", params.RoundID); err != nil {
		return err
	}
	if err := validateExactArtifact("round manifest", params.Manifest, false); err != nil {
		return err
	}
	if err := validateJSONObject("baseline snapshot", params.BaselineSnapshot, MaxSnapshotBytes); err != nil {
		return err
	}
	if params.DeadlineAt.IsZero() {
		return invalidf("Audit deadline is invalid")
	}
	if err := validateRoundItems(params.Items); err != nil {
		return err
	}
	if len(params.InitialRetained) > MaxArtifactLinksPerCall {
		return invalidf("initial retained artifact payload is too large")
	}
	if _, err := validateArtifactLinks(params.InitialRetained); err != nil {
		return err
	}
	return validateIdempotency(params.IdempotencyKey, params.RequestDigest)
}

func validateAcceptRound(params AcceptRoundParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if params.ExpectedAuditRevision == 0 || params.ExpectedAuditRevision > math.MaxInt64 || params.RoundOrdinal < 2 {
		return invalidf("next Round precondition is invalid")
	}
	if err := validateID("previousRoundID", params.PreviousRoundID); err != nil {
		return err
	}
	if err := validateID("roundID", params.RoundID); err != nil {
		return err
	}
	if params.RoundID == params.PreviousRoundID {
		return invalidf("next Round identity equals its predecessor")
	}
	if err := validateExactArtifact("round manifest", params.Manifest, false); err != nil {
		return err
	}
	if len(params.Items) == 0 {
		return invalidf("next Round must contain work")
	}
	if err := validateRoundItems(params.Items); err != nil {
		return err
	}
	seenSources := make(map[string]struct{}, len(params.Items))
	for _, item := range params.Items {
		if len(item.ProposalSources) != 1 {
			return invalidf("next Round item must have one exact proposal source")
		}
		for _, source := range item.ProposalSources {
			if err := validateID("proposal receipt ID", source.ReceiptID); err != nil {
				return err
			}
			if source.ProposedCheckOrdinal < 0 || source.ProposedCheckOrdinal >= 512 {
				return invalidf("proposal check ordinal is invalid")
			}
			if err := validateExactArtifact("proposal source", source.Proposal, true); err != nil {
				return err
			}
			key := source.ReceiptID + "\x00" + strconv.Itoa(source.ProposedCheckOrdinal)
			if _, duplicate := seenSources[key]; duplicate {
				return invalidf("proposal check source is duplicated")
			}
			seenSources[key] = struct{}{}
		}
	}
	return nil
}

func validateRoundItems(items []MaterializedItem) error {
	if len(items) > 10_000 {
		return invalidf("round has too many items")
	}
	seenIDs := make(map[string]struct{}, len(items))
	seenKeys := make(map[string]struct{}, len(items))
	materializedBytes := 0
	for index, item := range items {
		if err := validateID("itemID", item.ItemID); err != nil || item.Ordinal != index {
			return invalidf("round item %d identity or ordinal is invalid", index)
		}
		if _, duplicate := seenIDs[item.ItemID]; duplicate {
			return invalidf("round item ID is duplicated")
		}
		seenIDs[item.ItemID] = struct{}{}
		if err := validateText("item key", item.ItemKey, 256, true); err != nil {
			return err
		}
		if _, duplicate := seenKeys[item.ItemKey]; duplicate {
			return invalidf("round item key is duplicated")
		}
		seenKeys[item.ItemKey] = struct{}{}
		if err := validateText("item kind", item.Kind, 128, true); err != nil {
			return err
		}
		if err := validateText("item subject", item.SubjectKey, 512, true); err != nil {
			return err
		}
		if err := validateText("item Workflow role", item.WorkflowRole, 128, true); err != nil {
			return err
		}
		if item.InitialState != ItemReady && item.InitialState != ItemAwaitingReview {
			return invalidf("round item initial state is invalid")
		}
		approvalKind := item.ApprovalKind
		if approvalKind == "" {
			approvalKind = ItemApprovalNone
		}
		if !approvalKind.Valid() ||
			(approvalKind == ItemApprovalNone) != (item.ApprovalDigest == "") ||
			(item.InitialState == ItemAwaitingReview) != (approvalKind != ItemApprovalNone) {
			return invalidf("round item approval is invalid")
		}
		if item.ApprovalDigest != "" {
			if err := validateDigest("item approval digest", item.ApprovalDigest); err != nil {
				return err
			}
		}
		if err := validateExactArtifact("item task", item.Task, false); err != nil {
			return err
		}
		if err := validateItemOrigin(item.Origin, item.ItemKey, false); err != nil {
			return err
		}
		if err := validateCoverage(item.Coverage); err != nil || item.Coverage.Status != CoverageNotTested {
			return invalidf("initial item coverage is invalid")
		}
		encoded, err := json.Marshal(item)
		if err != nil || len(encoded) > MaxRoundPayloadBytes-materializedBytes {
			return invalidf("round materialization payload is too large")
		}
		materializedBytes += len(encoded)
	}
	return nil
}

func validateClaimIdentity(claim ControllerClaim) error {
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

func validateClaimParams(params ClaimParams) error {
	if err := validateText("claim holder", params.HolderID, 256, true); err != nil {
		return err
	}
	if params.Lease < time.Second || params.Lease > 5*time.Minute || params.Limit < 1 || params.Limit > MaxClaimBatch {
		return invalidf("claim bounds are invalid")
	}
	return nil
}

func validEventKind(value string) bool { return eventKindPattern.MatchString(value) }

func nonNilStrings(values []string) []string {
	return append([]string{}, values...)
}

func validateExecutionIntent(params CreateExecutionIntentParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if err := validateID("executionID", params.ExecutionID); err != nil {
		return err
	}
	if !params.Role.Valid() {
		return invalidf("execution role is invalid")
	}
	if err := validateText("execution Workflow role", params.WorkflowRole, 128, true); err != nil {
		return err
	}
	if params.Role == ExecutionCheck {
		if params.RoundID == nil || params.RoleAttempt != nil || len(params.Members) < 1 {
			return invalidf("check execution shape is invalid")
		}
		if err := validateID("roundID", *params.RoundID); err != nil {
			return err
		}
	} else {
		if params.RoleAttempt == nil || *params.RoleAttempt < 1 || len(params.Members) != 0 {
			return invalidf("non-check execution shape is invalid")
		}
		if params.RoundID != nil {
			if err := validateID("roundID", *params.RoundID); err != nil {
				return err
			}
		}
	}
	if len(params.Members) > MaxCollectionItems {
		return invalidf("execution batch is too large")
	}
	if err := validateExactArtifact("execution manifest", params.Manifest, false); err != nil {
		return err
	}
	if !idempotencyKeyPattern.MatchString(params.SubmissionKey) {
		return invalidf("submission key is invalid")
	}
	if err := validateDigest("execution request digest", params.RequestDigest); err != nil {
		return err
	}
	seenExecutionItems := make(map[string]struct{}, len(params.Members))
	seenItems := make(map[string]struct{}, len(params.Members))
	intentBytes := 0
	for index, member := range params.Members {
		if err := validateID("executionItemID", member.ExecutionItemID); err != nil ||
			validateID("itemID", member.ItemID) != nil || member.BatchOrdinal != index || member.ItemAttempt < 1 {
			return invalidf("execution member %d identity is invalid", index)
		}
		if _, exists := seenExecutionItems[member.ExecutionItemID]; exists {
			return invalidf("execution item ID is duplicated")
		}
		seenExecutionItems[member.ExecutionItemID] = struct{}{}
		if _, exists := seenItems[member.ItemID]; exists {
			return invalidf("execution item membership is duplicated")
		}
		seenItems[member.ItemID] = struct{}{}
		if err := validateExactArtifact("execution task", member.Task, false); err != nil {
			return err
		}
		if len(member.Inputs) > 128 {
			return invalidf("execution member has too many inputs")
		}
		for _, input := range member.Inputs {
			if err := validateExactArtifact("execution input", input, false); err != nil {
				return err
			}
		}
		encodedInputs, err := json.Marshal(member.Inputs)
		if err != nil || len(encodedInputs) > MaxExecutionInputsBytes ||
			len(encodedInputs) > MaxExecutionIntentBytes-intentBytes {
			return invalidf("execution inputs are too large")
		}
		intentBytes += len(encodedInputs)
	}
	return nil
}

func validateBindRun(params BindRunParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if err := validateID("executionID", params.ExecutionID); err != nil {
		return err
	}
	return validateID("runID", params.RunID)
}

func validateObserveTerminal(params ObserveTerminalParams) error {
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

func validateSubmissionFailure(params ObserveSubmissionFailureParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	return validateID("executionID", params.ExecutionID)
}

func validateCollect(params CollectParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if err := validateID("receiptID", params.ReceiptID); err != nil {
		return err
	}
	if err := validateID("executionID", params.ExecutionID); err != nil {
		return err
	}
	if !params.Disposition.Valid() {
		return invalidf("collection disposition is invalid")
	}
	if err := validateDigest("collection request digest", params.RequestDigest); err != nil {
		return err
	}
	if params.Disposition == CollectionAccepted || params.Disposition == CollectionInvalidResult {
		if params.SourceOutput == nil {
			return invalidf("accepted or invalid collection requires a source output")
		}
		if err := validateExactArtifact("collection source output", *params.SourceOutput, false); err != nil {
			return err
		}
	} else if params.SourceOutput != nil {
		return invalidf("collection disposition forbids a source output")
	}
	if params.ErrorCode != nil {
		if err := validateText("collection error code", *params.ErrorCode, 128, true); err != nil {
			return err
		}
	}
	if len(params.Items) > MaxCollectionItems || len(params.Retained) > MaxArtifactLinksPerCall {
		return invalidf("collection payload is too large")
	}
	seenItems := make(map[string]struct{}, len(params.Items))
	seenFindingReceipts := make(map[string]struct{})
	collectionBytes := 0
	for _, item := range params.Items {
		if err := validateID("executionItemID", item.ExecutionItemID); err != nil {
			return err
		}
		if _, duplicate := seenItems[item.ExecutionItemID]; duplicate {
			return invalidf("collection item is duplicated")
		}
		seenItems[item.ExecutionItemID] = struct{}{}
		if !item.Disposition.Valid() || !item.FinalDisposition.Valid() {
			return invalidf("collection item disposition is invalid")
		}
		if item.Disposition != params.Disposition ||
			item.FinalDisposition != finalForCollection(item.Disposition) {
			return invalidf("collection item disposition does not match its receipt")
		}
		if item.Disposition == CollectionAccepted {
			if item.Result == nil {
				return invalidf("accepted collection item requires a result")
			}
			if err := validateExactArtifact("collection item result", *item.Result, false); err != nil {
				return err
			}
			if item.Retryable || item.FinalDisposition != FinalAccepted {
				return invalidf("accepted collection item cannot retry")
			}
		} else if item.Result != nil {
			return invalidf("non-accepted collection item forbids a result")
		}
		if item.Disposition != CollectionAccepted && len(item.FindingAssociations) != 0 {
			return invalidf("non-accepted collection item forbids finding associations")
		}
		if len(item.FindingAssociations) > 128 {
			return invalidf("collection item has too many finding associations")
		}
		for _, association := range item.FindingAssociations {
			if err := validateID("finding assessment ID", association.AssessmentID); err != nil {
				return err
			}
			if err := validateID("finding receipt ID", association.ReceiptID); err != nil {
				return err
			}
			if _, duplicate := seenFindingReceipts[association.ReceiptID]; duplicate {
				return invalidf("finding receipt is associated more than once")
			}
			seenFindingReceipts[association.ReceiptID] = struct{}{}
			if err := validateExactArtifact("finding proposal", association.Proposal, true); err != nil {
				return err
			}
			switch association.SemanticAssessment {
			case "supported", "refuted", "inconclusive", "blocked", "satisfied", "violated", "not-tested":
			default:
				return invalidf("finding semantic assessment is invalid")
			}
		}
		if item.Disposition == CollectionExecutionCancelled && item.Retryable {
			return invalidf("cancelled collection item cannot retry")
		}
		if err := validateCoverage(item.Coverage); err != nil {
			return err
		}
		encoded, err := json.Marshal(item)
		if err != nil || len(encoded) > MaxCollectionBytes-collectionBytes {
			return invalidf("collection item payload is too large")
		}
		collectionBytes += len(encoded)
	}
	if _, err := validateArtifactLinks(params.Retained); err != nil {
		return err
	}
	return nil
}

func validateArtifactLinks(links []ArtifactLink) (int64, error) {
	seenLinks := make(map[string]struct{}, len(links))
	var retainedBytes int64
	retainedArtifacts := make(map[string]struct{}, len(links))
	for _, link := range links {
		if err := validateText("artifact logical key", link.LogicalKey, 512, true); err != nil {
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
		artifactKey := link.Artifact.Ref.Namespace + "\x00" + link.Artifact.Ref.Name + "\x00" + *link.Artifact.Ref.Revision
		if _, counted := retainedArtifacts[artifactKey]; !counted {
			if link.Artifact.SizeBytes > 1<<30-retainedBytes {
				return 0, invalidf("retained artifact bytes overflow")
			}
			retainedArtifacts[artifactKey] = struct{}{}
			retainedBytes += link.Artifact.SizeBytes
		}
	}
	encoded, err := json.Marshal(links)
	if err != nil || len(encoded) > MaxRetainedRefsBytes {
		return 0, invalidf("retained artifact refs are too large")
	}
	return retainedBytes, nil
}

func validateCommitReport(params CommitReportParams) error {
	if err := validateClaimIdentity(params.Claim); err != nil {
		return err
	}
	if params.ExpectedAuditRevision == 0 || params.ExpectedAuditRevision > math.MaxInt64 ||
		params.ExpectedRoundRevision == 0 || params.ExpectedRoundRevision > math.MaxInt64 {
		return invalidf("Audit report revision precondition is invalid")
	}
	if err := validateID("roundID", params.RoundID); err != nil {
		return err
	}
	if err := validateDigest("Audit report request digest", params.RequestDigest); err != nil {
		return err
	}
	for _, pair := range []struct {
		link  ArtifactLink
		key   string
		media string
	}{
		{params.Machine, ReportMachineLogicalKey, "application/json"},
		{params.Summary, ReportSummaryLogicalKey, "text/markdown"},
	} {
		legacySummary := pair.key == ReportSummaryLogicalKey && pair.link.Artifact.MediaType == "text/plain"
		if pair.link.LogicalKey != pair.key || (pair.link.Artifact.MediaType != pair.media && !legacySummary) {
			return invalidf("Audit report artifact contract is invalid")
		}
		if err := validateExactArtifact("Audit report artifact", pair.link.Artifact, true); err != nil {
			return err
		}
		if err := validateJSONObject("Audit report provenance", pair.link.SourceProvenance, 1<<20); err != nil {
			return err
		}
		if err := validateText("Audit report display ref", pair.link.DisplayRef, 1024, false); err != nil {
			return err
		}
	}
	return nil
}

func finalForCollection(value CollectionDisposition) FinalDisposition {
	switch value {
	case CollectionAccepted:
		return FinalAccepted
	case CollectionMissingOutput:
		return FinalMissingOutput
	case CollectionInvalidResult:
		return FinalInvalidResult
	case CollectionExecutionFailed:
		return FinalExecutionFailed
	case CollectionExecutionCancelled:
		return FinalExecutionCancelled
	case CollectionContractInvalid:
		return FinalInvalidResult
	default:
		return ""
	}
}
