package auditdomain

const (
	PackageMediaType = "application/zip"

	MaximumArchiveBytes     = 16 << 20
	MaximumExpandedBytes    = 32 << 20
	MaximumMembers          = 1024
	MaximumMemberBytes      = 16 << 20
	MaximumManifestBytes    = 512 << 10
	MaximumDocumentBytes    = 8 << 20
	MaximumGeneratedBytes   = 64 << 20
	MaximumPathBytes        = 512
	MaximumPathComponents   = 16
	MaximumIdentifierBytes  = 160
	MaximumDiagnosticBytes  = 256
	MaximumStringBytes      = 64 << 10
	MaximumItems            = 4096
	MaximumEvidencePerItem  = 256
	MaximumCoverageValues   = 512
	MaximumProposalsPerItem = 128
	MaximumJSONDepth        = 64
	MaximumJSONNodes        = 250000
	MaximumReferenceDepth   = 32
)
