package auditdomain

const (
	PackageMediaType = "application/zip"

	MaximumArchiveBytes    = 16 << 20
	MaximumExpandedBytes   = 32 << 20
	MaximumMembers         = 1024
	MaximumMemberBytes     = 16 << 20
	MaximumManifestBytes   = 512 << 10
	MaximumDocumentBytes   = 8 << 20
	MaximumGeneratedBytes  = 64 << 20
	MaximumPathBytes       = 512
	MaximumPathComponents  = 16
	MaximumIdentifierBytes = 160
	MaximumDiagnosticBytes = 256
	MaximumStringBytes     = 64 << 10
	MaximumItems           = 4096
	MaximumEvidencePerItem = 256
	MaximumCoverageValues  = 512
	MaximumJSONDepth       = 64
	MaximumJSONNodes       = 250000
	MaximumReferenceDepth  = 32
)

// MaximumCoverageValueBytes bounds both admitted proposal limitations and
// retained Audit coverage values.
const MaximumCoverageValueBytes = 512

// MaximumCollectionTransactionWork bounds the work of one transaction that
// retains collection material of an Audit child Run (LIM-AUX-7). Finding
// intake counts a proposal receipt once for its receipt, hold and
// direct-verification work, plus once for each proposal or evidence revision
// it imports, and once in all when the receipt was already retained. The
// importer counts each retained result, evidence or role output revision
// once. A transaction always retains at least one proposal or revision.
const MaximumCollectionTransactionWork = 512
