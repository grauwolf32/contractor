package auditcontroller

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func (c *Controller) deleteReleasedRunBatch(
	ctx context.Context, claim auditstore.ControllerClaim, ownerID string,
) (bool, bool, error) {
	batch := c.newDrainBatch()
	for batch.open(ctx) {
		runID, found, err := c.store.NextLiveRunForDeletion(ctx, claim)
		if err != nil {
			deleted, err := batch.stop(ctx, err)
			return deleted, false, err
		}
		if !found {
			return batch.committed > 0, true, nil
		}
		err = c.deleter.DeleteReleasedTerminalRun(ctx, ownerID, runID)
		var blocked *rundeletion.NotDeletableError
		if errors.As(err, &blocked) || errors.Is(err, runstore.ErrNotFound) {
			break
		}
		if err != nil {
			deleted, err := batch.stop(ctx, err)
			return deleted, false, err
		}
		batch.commit()
	}
	return batch.committed > 0, false, nil
}
