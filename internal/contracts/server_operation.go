package contracts

// Registered Server operation identities are shared by configuration and
// operation composition. Their implementations live outside this contract.
const (
	MemorySnapshotOperation      = "memory-snapshot@1"
	PublishFindingsOperation     = "publish-findings@1"
	SitemapOpenAPIJoinOperation  = "sitemap-openapi-join@1"
	SitemapApplyOpenAPIOperation = "sitemap-apply-openapi@1"
	OpenAPIPrepareWorkOperation  = "openapi-prepare-work@1"
)
