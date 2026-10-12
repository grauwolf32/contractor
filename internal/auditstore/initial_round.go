package auditstore

const InitialInventoryLogicalKey = "initial-inventory"

// AcceptInitialRoundParams contains only derived state. The original baseline,
// deadline and preparation receipts remain unchanged when it is accepted.
type AcceptInitialRoundParams struct {
	Claim                 ControllerClaim
	ExpectedAuditRevision uint64
	RoundID               string
	Manifest              ExactArtifact
	Items                 []MaterializedItem
	Inventory             ArtifactLink
}
