// Package runstoretest holds the WorkflowRun setup that tests of other
// packages share. It stores Runs only through the RunStore's production
// paths; production code never imports it.
package runstoretest

import (
	"context"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// CreateRun stores params as a new initializing WorkflowRun through the
// public create-Run path, CreateRunIdempotent, with an idempotency key and a
// request digest derived from the Run ID. A Run ID that is already stored is
// runstore.ErrConflict, whichever owner stored it.
func CreateRun(
	ctx context.Context, store *runstore.PostgresStore, params runstore.CreateRunParams,
) (runstore.WorkflowRun, error) {
	digest := contentdigest.Bytes([]byte("runstoretest.CreateRun\x00" + params.RunID))
	run, inserted, err := store.CreateRunIdempotent(ctx, runstore.CreateRunIdempotentParams{
		CreateRunParams: params,
		IdempotencyKey:  "test-" + strings.TrimPrefix(digest, "sha256:"),
		RequestDigest:   digest,
	})
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	if !inserted {
		return runstore.WorkflowRun{}, fmt.Errorf("create WorkflowRun %q: %w", params.RunID, runstore.ErrConflict)
	}
	return run, nil
}
