package resolved

// FindingCollectionMediaType identifies a frozen finding collection ZIP. The
// Audit domain builds and publishes the package; the security-findings@1
// descriptor declares the Workflow input of its list_findings reader that must
// accept it.
const FindingCollectionMediaType = "application/vnd.contractor.findings-collection+zip"
