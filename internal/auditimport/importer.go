package auditimport

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// Collect retains ordinary outputs under one natural receipt without changing
// the child Run outcome. The existing claim fences Artifact writes and commit.
func (i *Importer) Collect(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	snapshot auditstore.ReconcileSnapshot,
	execution auditstore.Execution,
) (bool, error) {
	bound := *i
	bound.artifacts = i.artifacts.WithClaim(claim)
	i = &bound
	return i.collectTasks(ctx, claim, snapshot, execution)
}
