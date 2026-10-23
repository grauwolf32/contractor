package apierror

import (
	"fmt"
	"net/http"
)

// registry holds every entry in definition order; byCode indexes it.
var (
	registry []*entry
	byCode   = map[string]Kind{}
)

// define registers one code. Defining a code twice is a programming error.
func define(code string, status int, retryable bool, message, meaning, owner string) Kind {
	return register(&entry{
		code: code, status: status, retryable: retryable,
		message: message, meaning: meaning, owner: owner,
	})
}

// defineFrame registers a code that a WebSocket error frame can carry too.
func defineFrame(code string, status int, retryable bool, message, meaning, owner string) Kind {
	return register(&entry{
		code: code, status: status, retryable: retryable, frame: true,
		message: message, meaning: meaning, owner: owner,
	})
}

func register(row *entry) Kind {
	if _, exists := byCode[row.code]; exists {
		panic(fmt.Sprintf("apierror: code %q is defined twice", row.code))
	}
	registry = append(registry, row)
	kind := Kind{row}
	byCode[row.code] = kind
	return kind
}

const (
	publicAPI   = "../api/public-api.md"
	runtimeCfg  = "../configuration/runtime-config.md"
	gitImport   = "../artifacts/git-import.md"
	managedEval = "../evals/managed-evals.md"
)

func link(id, document string) string { return "[" + id + "](" + document + ")" }

// Request, authentication and transport failures shared by every operation.
var (
	InvalidRequest = define("invalid_request", http.StatusBadRequest, false,
		"request does not satisfy the API contract",
		"The request does not satisfy the API contract: a malformed, unknown or duplicate member, parameter or header, "+
			"a wrong media type, a domain validation failure, a page cursor reused with another query or snapshot, "+
			"an out-of-range performance history query or a WebSocket handshake that is not a valid upgrade; nothing changes.",
		link("API-R71", publicAPI)+", "+link("CAT-R68", "../configuration/catalog.md")+", "+
			link("OPS-R60", "../operations/operations.md")+", "+link("OPS-R165", "../operations/operations.md"))
	Unauthorized = defineFrame("unauthorized", http.StatusUnauthorized, false,
		"valid authentication is required",
		"Authentication is absent or invalid: no valid bearer token or browser session, or a login with invalid credentials.",
		link("SEC-R45", "security.md")+", "+link("API-R33", publicAPI))
	Forbidden = define("forbidden", http.StatusForbidden, false,
		"request is not permitted",
		"The authenticated principal lacks the Operations capability, or a browser request failed its Origin, "+
			"CSRF or CORS preflight check.",
		link("SEC-R45", "security.md")+", "+link("API-R23", publicAPI)+", "+link("OPS-R60", "../operations/operations.md"))
	NotFound = defineFrame("not_found", http.StatusNotFound, false,
		"resource was not found",
		"The resource does not exist or is not visible to the principal, and the answer does not disclose which: "+
			"ownership checks use the same 404 for another owner's resource, and a Runtime credential the principal "+
			"may not use is reported like a missing one. As a WebSocket error frame, the subscribed stream or "+
			"subscription is not found.",
		link("API-R25", publicAPI)+", "+link("PRJ-R75", "../artifacts/projects.md"))
	MethodNotAllowed = define("method_not_allowed", http.StatusMethodNotAllowed, false,
		"method is not allowed",
		"The path belongs to a public operation but not with this method, including HEAD where the operation defines none.",
		link("API-R73", publicAPI))
	RequestTooLarge = define("request_too_large", http.StatusRequestEntityTooLarge, false,
		"request body exceeds its size limit",
		"A public JSON request body exceeds 1 MiB (LIM-API-1) or the smaller bound its resource declares; "+
			"rejected before the handler's validation.",
		link("API-R4", publicAPI))
	UnsupportedMediaType = define("unsupported_media_type", http.StatusUnsupportedMediaType, false,
		"Content-Type must be application/json",
		"The Scheduler settings update is not sent as application/json; nothing changes. Other operations answer "+
			"a wrong media type with invalid_request.",
		link("OPS-R60", "../operations/operations.md"))
	RateLimited = define("rate_limited", http.StatusTooManyRequests, true,
		"request is temporarily limited",
		"Failed login attempts exceeded the per-client-IP or process-wide limiter (the answer carries a bounded "+
			"Retry-After and creates no account lockout), or the browser session already holds its maximum of event "+
			"sockets (LIM-API-10).",
		link("SEC-R51", "security.md")+", "+link("API-R62", publicAPI))
	PreconditionFailed = define("precondition_failed", http.StatusPreconditionFailed, false,
		"resource revision precondition failed",
		"A strong If-Match revision is not the current revision of the resource (Project, Audit, Run Queue control, "+
			"Scheduler settings, RuntimeConfig binding, Agent labels); nothing changes, and the client re-reads "+
			"before deciding again.",
		link("API-R26", publicAPI)+", "+link("OPS-R57", "../operations/operations.md")+", "+link("RCF-R175", runtimeCfg))
	Conflict = define("conflict", http.StatusConflict, true,
		"resource state changed; retry with the current revision",
		"The resource state changed or an identity is taken: a stale Artifact CAS precondition, a Run, Project or "+
			"Audit state that no longer admits the request, or a duplicate identity. Re-read the current state and "+
			"retry with its revision.",
		link("API-R26", publicAPI)+", "+link("ART-R59", "../artifacts/artifact-store.md"))
	IdempotencyConflict = define("idempotency_conflict", http.StatusConflict, false,
		"Idempotency-Key is already bound to a different request; use a new key for a new request",
		"An Idempotency-Key already bound to a different request was sent again with another request (any change "+
			"to the bound body, path or precondition), or a finding collection clientKey already bound to another "+
			"selection. Permanent: the original result is unchanged and only a new key can carry the changed request; "+
			"the same request still replays. Managed Eval mutations report this outcome as eval_idempotency_conflict.",
		link("API-R69", publicAPI)+", "+link("FND-R114", "../findings/findings.md"))
	StorageTransactionConflict = define("storage_transaction_conflict", http.StatusServiceUnavailable, true,
		"database transaction conflicted; retry the request",
		"A definite PostgreSQL transaction conflict (serialization failure or deadlock), including one that outlasts "+
			"the bounded transaction retries of its operation; nothing committed, so the request may be repeated.",
		link("PER-R16", "persistence.md"))
	ServerStandby = define("server_standby", http.StatusServiceUnavailable, true,
		"Server is standing by for the active Server; retry shortly",
		"This Server is a standby: another Server holds the database's single-active lease, so it answers every "+
			"public API request, the event upgrade included, without authenticating or routing it, until it takes "+
			"over or the client reaches the active Server; nothing changes.",
		link("PER-R3", "persistence.md"))
	InternalError = define("internal_error", http.StatusInternalServerError, true,
		"request could not be processed",
		"An unclassified failure, answered without its cause. The Server cannot tell a transient failure from a "+
			"defect here, so the code is retryable and clients bound their retries.",
		link("API-R73", publicAPI))
)

// WebSocket error frames.
var (
	InvalidFrame = defineFrame("invalid_frame", 0, false,
		"frame does not satisfy the protocol",
		"A WebSocket client frame is malformed, unknown or reuses an active subscription ID; the socket and its "+
			"subscriptions stay unchanged.",
		link("API-R71", publicAPI)+", "+link("API-R43", publicAPI))
	SubscriptionLimit = defineFrame("subscription_limit", 0, false,
		"subscription limit reached",
		"A socket already holds its maximum of subscriptions (LIM-API-11); only the new subscription is refused.",
		link("API-R62", publicAPI)+", "+link("API-R63", publicAPI))
	Overloaded = defineFrame("overloaded", 0, true,
		"stream is temporarily unavailable",
		"The Server cannot currently read the subscribed Run or Operations stream; the subscription ends and the "+
			"client resubscribes after its backoff.",
		link("API-R64", publicAPI)+", "+link("API-R72", publicAPI))
)

// Artifacts and Projects.
var (
	ArtifactTooLarge = define("artifact_too_large", http.StatusRequestEntityTooLarge, false,
		"artifact exceeds the 64 MiB limit",
		"An Artifact exceeds its 64 MiB limit (LIM-ART-4); rejected in the same way as an oversized request body.",
		link("API-R5", publicAPI))
	ArtifactFrozen = define("artifact_frozen", http.StatusConflict, false,
		"artifact binding is frozen and cannot change",
		"A write targets a frozen binding, which never advances, so neither a re-read nor a retry can succeed; "+
			"returned on the public and private Artifact APIs whatever the write precondition.",
		link("ART-R59", "../artifacts/artifact-store.md"))
	ArtifactTransferCapacity = define("artifact_transfer_capacity", http.StatusServiceUnavailable, true,
		"artifact transfer capacity is exhausted",
		"All full-payload transfer slots of the Server process are taken; the request fails promptly instead of "+
			"queueing its body.",
		link("BLB-R25", "../artifacts/blob-storage.md"))
	ArtifactContentCorrupt = define("artifact_content_corrupt", http.StatusServiceUnavailable, false,
		"artifact content failed integrity verification",
		"Referenced payload content failed its digest or size verification, or an exact binding holds other content; "+
			"no retry repairs it.",
		link("BLB-R46", "../artifacts/blob-storage.md"))
	ArtifactContentMissing = define("artifact_content_missing", http.StatusServiceUnavailable, false,
		"artifact content is unavailable",
		"Referenced payload content is missing, or a live read lost the race with registry deletion; never an "+
			"empty success, a silent revision replacement or an automatic Git refetch.",
		link("BLB-R46", "../artifacts/blob-storage.md"))
	ProjectDeleting = define("project_deleting", http.StatusConflict, false,
		"Project deletion is in progress",
		"The Project, or the scope of the Run, Audit or Artifact the request names, is being deleted; the request is "+
			"refused for good.",
		link("DEL-R40", "../artifacts/retention-and-deletion.md"))
	RunNotDeletable = define("run_not_deletable", http.StatusConflict, true,
		"Run cannot be deleted yet",
		"DELETE /v1/runs/{runId} is refused because the Run is not terminal (run_not_terminal), its allocations "+
			"have not completed release (allocation_release_pending) or an Audit-bound Run has no durable "+
			"AuditCollectionReceipt (audit_collection_pending); nothing changes. Retry after a refetch shows the Run "+
			"deletable.",
		link("DEL-R20", "../artifacts/retention-and-deletion.md"))
)

// Archive previews.
var (
	ArchiveMediaType = define("archive_media_type", http.StatusUnsupportedMediaType, false,
		"Archive preview requires a ZIP or Skill package",
		"An archive preview (getArtifactArchive and its Project and Run variants) names an Artifact that is not a "+
			"ZIP or Skill package.",
		link("API-R73", publicAPI))
	ArchiveInvalid = define("archive_invalid", http.StatusUnprocessableEntity, false,
		"Archive is invalid, unsupported or contains unsafe entries. Download the original to inspect it separately.",
		"The previewed archive is malformed, unsupported or contains unsafe entries; the original stays downloadable.",
		link("API-R73", publicAPI))
	ArchivePreviewLimit = define("archive_preview_limit", http.StatusUnprocessableEntity, false,
		"Archive preview limit exceeded. Text files must fit 1 MiB; the original archive remains downloadable.",
		"The archive exceeds a preview bound, or a previewed file exceeds 1 MiB of text or 4 MiB of compressed input.",
		link("API-R73", publicAPI))
	ArchiveFileNotText = define("archive_file_not_text", http.StatusUnprocessableEntity, false,
		"This file is not supported UTF-8 text. Download the original archive to view it separately.",
		"A previewed archive file is not UTF-8 text.",
		link("API-R73", publicAPI))
)

// Git import and the owner Git key.
var (
	GitURLInvalid = define("git_url_invalid", http.StatusBadRequest, false,
		"Git repository URL is invalid",
		"The repository URL is not an accepted https:// or SSH form, or carries a password, query or fragment.",
		link("GIT-R13", gitImport)+", "+link("GIT-R15", gitImport)+", "+link("GIT-R86", gitImport))
	GitRefUnavailable = define("git_ref_unavailable", http.StatusUnprocessableEntity, false,
		"Git branch or tag is missing or ambiguous",
		"The requested branch or tag is missing or an ambiguous short name.",
		link("GIT-R17", gitImport)+", "+link("GIT-R18", gitImport))
	GitDestinationRefused = define("git_destination_refused", http.StatusUnprocessableEntity, false,
		"Git remote destination is not allowed",
		"The remote is outside the operator allowlist or resolves to a refused destination.",
		link("GIT-R52", gitImport)+", "+link("GIT-R56", gitImport))
	GitHostUntrusted = define("git_host_untrusted", http.StatusUnprocessableEntity, false,
		"Git SSH host trust is unavailable or verification failed",
		"SSH host verification failed closed: an unknown or changed host key, or no known_hosts configuration.",
		link("GIT-R49", gitImport)+", "+link("GIT-R50", gitImport)+", "+link("GIT-R51", gitImport))
	GitRemoteUnavailable = define("git_remote_unavailable", http.StatusUnprocessableEntity, false,
		"Git repository is unavailable or access was refused",
		"The repository is unavailable, refuses access, or cannot provide a bounded snapshot of the selected ref.",
		link("GIT-R86", gitImport))
	GitContentUnsupported = define("git_content_unsupported", http.StatusUnprocessableEntity, false,
		"Git snapshot contains unsupported or invalid content",
		"The tree has unsafe or unsupported content, no readable source, or an analysis report path collision. "+
			"Complete snapshots also reject symlinks, submodules and Git LFS pointers, naming the first entry. Nothing is published.",
		link("GIT-R25", gitImport)+", "+link("GIT-R28", gitImport))
	GitImportLimit = define("git_import_limit", http.StatusRequestEntityTooLarge, false,
		"Git import exceeds a resource limit",
		"The import exceeds one of its budgets (LIM-GIT); nothing is published.",
		link("GIT-R66", gitImport))
	GitImportCapacity = define("git_import_capacity", http.StatusServiceUnavailable, false,
		"Another Git import is active",
		"The Server process already runs its one admitted Git import (LIM-GIT-19); the request is refused promptly "+
			"without queueing. The client starts a new import once the active one ends.",
		link("GIT-R69", gitImport)+", "+link("GIT-R70", gitImport))
	GitImportUnavailable = define("git_import_unavailable", http.StatusServiceUnavailable, false,
		"Git import is unavailable",
		"Git import is not configured on this Server: the remote allowlist or known_hosts configuration is missing "+
			"or invalid.",
		link("GIT-R53", gitImport)+", "+link("GIT-R54", gitImport))
	GitImportTimeout = define("git_import_timeout", http.StatusGatewayTimeout, false,
		"Git import timed out; inspect artifact metadata before retrying",
		"The whole-import deadline (LIM-GIT-6) expired; nothing partial is published, and a lost success is "+
			"resolved through the Artifact metadata before importing again.",
		link("GIT-R72", gitImport)+", "+link("GIT-R73", gitImport)+", "+link("GIT-R80", gitImport))
	GitImportCancelled = define("git_import_cancelled", http.StatusRequestTimeout, false,
		"Git import was interrupted; inspect artifact metadata before retrying",
		"The request was cancelled during the import; nothing partial is published, and a racing commit is "+
			"resolved through the Artifact metadata.",
		link("GIT-R73", gitImport)+", "+link("GIT-R79", gitImport))
	GitKeyInvalid = define("git_key_invalid", http.StatusBadRequest, false,
		"Git SSH key must be a supported unencrypted private key of at most 32 KiB",
		"The owner Git key is not a supported unencrypted private key of at most 32 KiB.",
		link("GIT-R39", gitImport)+", "+link("GIT-R40", gitImport))
	GitKeyMissing = define("git_key_missing", http.StatusUnprocessableEntity, false,
		"Git SSH key is not configured",
		"An SSH import needs the owner's Git key, which is not configured.",
		link("GIT-R38", gitImport))
	GitKeyUnavailable = define("git_key_unavailable", http.StatusServiceUnavailable, false,
		"Git key encryption is not configured",
		"The Server has no credential master key, so it can neither store nor use an owner Git key.",
		link("GIT-R41", gitImport)+", "+link("GIT-R86", gitImport))
)

// Catalog publication and managed credentials.
var (
	ConfigurationConflict = define("configuration_conflict", http.StatusConflict, false,
		"immutable configuration identity or idempotency key conflicts",
		"A managed publication reused a published (kind, name, version) identity with different normalized "+
			"content; an identical identity and body instead returns the published version.",
		link("CAT-R59", "../configuration/catalog.md"))
	ConfigurationDrift = define("configuration_drift", http.StatusConflict, false,
		"configuration roots changed outside the publication manager; restore them or restart the Server",
		"A managed publication found that the reloaded configuration roots differ from the Server's loaded "+
			"snapshot by a change the Server did not make; nothing is published, and publication resumes once the "+
			"roots match the loaded snapshot again or after a restart.",
		link("CAT-R75", "../configuration/catalog.md"))
	CredentialConflict = define("credential_conflict", http.StatusConflict, false,
		"credential identity or idempotency key conflicts",
		"An LLM credential creation names an identity that already exists.",
		link("RCF-R41", runtimeCfg))
	CredentialInUse = define("credential_in_use", http.StatusConflict, false,
		"credential is referenced by an active Run, Audit or RuntimeConfig label",
		"LLM credential deletion is refused because a non-terminal Run, active RuntimeConfig label binding or live "+
			"allocation pins or references the credential; the details list the safe referencing IDs and nothing changes.",
		link("RCF-R48", runtimeCfg))
	GatewayUnavailable = define("gateway_unavailable", http.StatusBadGateway, true,
		"managed Gateway operation is unavailable",
		"The bound Gateway credential manager failed or is not configured, so the credential operation did not "+
			"complete; the database record is retained.",
		link("RCF-R49", runtimeCfg))
	CredentialKeyUnavailable = define("credential_key_unavailable", http.StatusServiceUnavailable, false,
		"credential master key is not configured on this Server",
		"The Server was started without a credential master key, so it can neither create, delete nor use an "+
			"LLM or Runtime credential; nothing is written and no Gateway call is made. No retry succeeds until an "+
			"operator configures the key and restarts the Server.",
		link("RCF-R44", runtimeCfg))
	CredentialRecoveryRequired = define("credential_recovery_required", http.StatusServiceUnavailable, true,
		"credential operation recovery is in progress",
		"An earlier credential operation ended ambiguously, so new Run initialization and credential changes wait "+
			"until the same request or startup recovery resolves it.",
		link("RCF-R53", runtimeCfg))
)

// Runtime configuration.
var (
	RuntimeConfigInvalid = define("runtime_config_invalid", http.StatusBadRequest, false,
		"RuntimeConfig request is invalid",
		"A RuntimeConfig, its resolved configuration, a selected RuntimeConfig version or a Runtime credential "+
			"request fails validation.",
		link("RCF-R178", runtimeCfg))
	RuntimeConfigConflict = define("runtime_config_conflict", http.StatusConflict, false,
		"Runtime configuration conflicts",
		"Different normalized values for one merge unit within one layer (Run-selected or Agent labels), never "+
			"resolved by label order, or a Runtime credential identity that already exists.",
		link("RCF-R151", runtimeCfg)+", "+link("RCF-R178", runtimeCfg))
	RuntimeLabelUnknown = define("runtime_label_unknown", http.StatusBadRequest, false,
		"a selected Runtime label is unavailable",
		"A Runtime label without a binding was selected or assigned.",
		link("RCF-R178", runtimeCfg)+", "+link("RUN-R85", "../execution/runs.md"))
	RuntimeLabelInUse = define("runtime_label_in_use", http.StatusConflict, false,
		"Runtime label is assigned to a Runtime Agent or reserved",
		"A Runtime label binding cannot be removed while the label is assigned to a Runtime Agent (the details "+
			"list the Agents), or the label is the reserved default.",
		link("RCF-R178", runtimeCfg)+", "+link("RCF-R129", runtimeCfg))
	RuntimeCredentialInUse = define("runtime_credential_in_use", http.StatusConflict, false,
		"Runtime credential is referenced by active configuration",
		"A Runtime credential cannot be deleted while an active label binding, Project, non-terminal Run or Audit "+
			"snapshot or live allocation references it; the details list the references.",
		link("RCF-R178", runtimeCfg))
	RuntimeAgentLabelNotApplicable = define("runtime_agent_label_not_applicable", http.StatusBadRequest, false,
		"Runtime label has no Worker-applicable setting",
		"An Agent label whose current RuntimeConfig has no Worker setting was assigned.",
		link("RCF-R143", runtimeCfg))
	RuntimeAgentInUse = define("runtime_agent_in_use", http.StatusConflict, false,
		"Runtime Agent principal is live, labeled, or allocation-referenced",
		"A Runtime Agent principal cannot be deleted while it is live, still has labels or an allocation "+
			"references it.",
		link("RCF-R178", runtimeCfg))
)

// Audits and their packages.
var (
	AuditProfileUnsupported = define("audit_profile_unsupported", http.StatusUnprocessableEntity, false,
		"AuditProfile requires unsupported Server capabilities",
		"The pinned AuditProfile requires Server capabilities that are missing; bounded reason codes in the "+
			"details name them and do not describe transient Worker availability.",
		link("AUD-R88", "../audits/audits.md"))
	AuditDocumentInvalid   = auditPackage("audit_document_invalid", "an Audit document is invalid")
	AuditSchemaUnsupported = auditPackage("audit_schema_unsupported",
		"an Audit document has an unsupported schema version")
	AuditLimitExceeded      = auditPackage("audit_limit_exceeded", "an Audit document or package exceeds a bound")
	AuditPackageInvalid     = auditPackage("audit_package_invalid", "an Audit package is invalid")
	AuditPackagePathInvalid = auditPackage("audit_package_path_invalid",
		"an Audit package member path is invalid")
	AuditPackageMemberForbidden = auditPackage("audit_package_member_forbidden",
		"an Audit package contains a forbidden member")
	AuditDigestMismatch = auditPackage("audit_digest_mismatch",
		"an Audit document or package member does not match its digest")
	AuditReferenceInvalid = auditPackage("audit_reference_invalid", "an Audit reference is invalid")
	AuditRemoteReference  = auditPackage("audit_remote_reference",
		"an Audit document references remote content")
	AuditInventoryInvalid = auditPackage("audit_inventory_invalid", "an Audit inventory is invalid")
	AuditResultSetInvalid = auditPackage("audit_result_set_invalid", "an Audit result set is invalid")
)

// auditPackage registers one Audit package codec code.
func auditPackage(code, failure string) Kind {
	return define(code, http.StatusUnprocessableEntity, false,
		"Audit input could not be processed",
		"Bounded Audit package codec error: "+failure+". The Audit input, finding proposal or finding collection is "+
			"rejected without its source bytes; in list_findings the Runtime tool transport maps it to its own error.",
		"[Audits](../audits/audits.md), [findings](../findings/findings.md)")
}

// Managed Evals. Every Eval failure is answered with retryable false: its
// recovery action in the details tells the client what to do instead.
var (
	EvalInvalid = evalCode("eval_invalid", http.StatusUnprocessableEntity, "Invalid evaluation document.",
		"An Eval request or document is invalid.", link("EVM-R229", managedEval))
	EvalLimitExceeded = evalCode("eval_limit_exceeded", http.StatusUnprocessableEntity,
		"Evaluation document exceeds format bounds.",
		"An Eval document or a native experiment matrix exceeds its bound: above 1,000 expanded members or a "+
			"conservative serialized-plan estimate above 1 MiB.",
		link("EVM-R63", managedEval)+", "+link("API-R6", publicAPI))
	EvalNotFound = evalCode("eval_not_found", http.StatusNotFound, "Evaluation resource not found.",
		"The Eval resource does not exist or is not visible to the principal.", link("EVM-R229", managedEval))
	EvalPreconditionRequired = evalCode("eval_precondition_required", http.StatusPreconditionRequired,
		"An evaluation revision is required.",
		"A revision-guarded Eval mutation was sent without its If-Match revision.", link("EVM-R35", managedEval))
	EvalRevisionMismatch = evalCode("eval_revision_mismatch", http.StatusPreconditionFailed,
		"The evaluation revision has changed.",
		"The If-Match revision is not the current experiment revision.", link("EVM-R36", managedEval))
	EvalIdempotencyConflict = evalCode("eval_idempotency_conflict", http.StatusConflict,
		"The operation key is already bound to another request.",
		"The Eval form of idempotency_conflict: the key is bound to another request, and only a new key can "+
			"carry the changed one (recovery retry_same_request).", link("API-R69", publicAPI))
	EvalNotReady = evalCode("eval_not_ready", http.StatusConflict, "The experiment is not ready.",
		"A command that the experiment's state rules out, such as a stop that progress has since ruled out; it "+
			"passed the ETag check, and the browser refetches instead of replaying it.", link("EVM-R37", managedEval))
	EvalPreparationUnavailable = evalCode("eval_preparation_unavailable", http.StatusServiceUnavailable,
		"Experiment preparation is temporarily unavailable.",
		"An unknown infrastructure failure during Prepare: the experiment reports it as its diagnostic with "+
			"recovery wait while the coordinator retries Prepare by itself.", link("EVM-R90", managedEval))
	EvalPinMismatch = evalCode("eval_pin_mismatch", http.StatusConflict, "Required evaluation pins do not match.",
		"A stop names another plan than the current one, or a required pin differs; a new plan is needed "+
			"(recovery duplicate).", link("EVM-R37", managedEval))
	EvalExternalControl = evalCode("eval_external_control", http.StatusConflict,
		"The experiment has a different execution controller.",
		"The request needs another execution controller (native or external) than the experiment has.",
		link("EVM-R227", managedEval))
	EvalViewChanged = evalCode("eval_view_changed", http.StatusConflict, "The selected evaluation view has changed.",
		"The view snapshot bound to a page or chart request changed; the client keeps its filters and restarts "+
			"pagination.", link("EVM-R215", managedEval))
	EvalMemberConflict = evalCode("eval_member_conflict", http.StatusConflict,
		"The evaluation member has conflicting records.",
		"Member records conflict, such as two executions claiming one member before receipts adjudicate it.",
		link("EVM-R227", managedEval))
	EvalEvidenceUnavailable = evalCode("eval_evidence_unavailable", http.StatusConflict,
		"Required evaluation evidence is unavailable.",
		"Evidence that the request needs is unavailable or was deleted.", link("EVM-R227", managedEval))
	EvalProducerStale = evalCode("eval_producer_stale", http.StatusConflict,
		"The external producer has no recent observation.",
		"An external producer's observation is too old for the request.", link("EVM-R227", managedEval))
	EvalBudgetExhausted = evalCode("eval_budget_exhausted", http.StatusConflict,
		"The frozen experiment allowance is exhausted.",
		"The experiment's frozen allowance is exhausted; a new plan is needed (recovery duplicate).",
		link("EVM-R227", managedEval))
	EvalProjectDeleting = evalCode("eval_project_deleting", http.StatusConflict,
		"The evaluation workspace is being deleted.",
		"The evaluation Project is being deleted, so the request is refused (recovery wait); an external "+
			"producer returning after the deletion fence cannot recreate deleted attempts.",
		link("EVM-R227", managedEval)+", "+link("DEL-R43", "../artifacts/retention-and-deletion.md"))
)

// evalCode registers one managed Eval code.
func evalCode(code string, status int, message, meaning, owner string) Kind {
	return define(code, status, false, message, meaning+" Retryable false: the recovery action in the details "+
		"says what to do.", owner)
}

// Finding correlation reads use explicit stale traversal and bounded-content errors.
var (
	FindingCorrelationStale = define("finding_correlation_stale", http.StatusConflict, false,
		"finding comparison changed; restart from the first page",
		"A selected Project or Audit revision changed during a candidate, comparison or member traversal; restart without mixing revisions.",
		link("FND-R179", "../findings/findings.md"))
	FindingCorrelationItemTooLarge = define("finding_correlation_item_too_large", http.StatusUnprocessableEntity, false,
		"finding comparison item cannot fit its bounded preview",
		"One mandatory comparison item cannot fit the bounded page representation; no observation is silently omitted.",
		link("FND-R179", "../findings/findings.md"))
)
