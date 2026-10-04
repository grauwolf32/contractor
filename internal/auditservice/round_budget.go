package auditservice

import "github.com/grauwolf32/contractor/internal/auditdomain"

// roundItemCapacity is the one per-Round item budget that Start, input
// preview and next-Round selection share: the profile's per-Round limit and
// the Audit's remaining total, bounded by the inventory schema maximum.
func roundItemCapacity(maxItemsPerRound, maxItemsTotal, usedItems int) int {
	return max(0, min(maxItemsPerRound, maxItemsTotal-usedItems, auditdomain.MaximumItems))
}
