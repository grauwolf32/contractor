package scheduler

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// TestPostgresRequiredWorkflowOutputsReadTheArtifactStore checks the output
// verification of a succeeding Run on the real store, where it reads the
// current revision of each output binding through the ArtifactStore: a bound
// output with an accepted media type passes, a missing optional one is
// skipped, and a missing required one or an incompatible media type fails.
func TestPostgresRequiredWorkflowOutputsReadTheArtifactStore(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-outputs", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Outputs",
		IdempotencyKey: "project-outputs", RequestDigest: "sha256:" + strings.Repeat("a", 64),
	}); err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	createBoundProjectRunOutput(t, ctx, pool, service, "run-outputs", "project-outputs", "result", "bound output")

	text := []string{"text/plain"}
	for _, test := range []struct {
		name  string
		slots map[string]resolved.ArtifactSlot
		want  string
	}{
		{name: "bound", slots: map[string]resolved.ArtifactSlot{"result": {Required: true, MediaTypes: text}}},
		{name: "optional missing", slots: map[string]resolved.ArtifactSlot{
			"result": {Required: true, MediaTypes: text}, "notes": {MediaTypes: text},
		}},
		{name: "required missing", slots: map[string]resolved.ArtifactSlot{
			"result": {Required: true, MediaTypes: text}, "summary": {Required: true, MediaTypes: text},
		}, want: `required Workflow output "summary" is not bound`},
		{name: "incompatible media type", slots: map[string]resolved.ArtifactSlot{
			"result": {Required: true, MediaTypes: []string{"application/json"}},
		}, want: `Workflow output "result" has incompatible media type`},
	} {
		t.Run(test.name, func(t *testing.T) {
			var verified error
			if err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
				verified = verifyRequiredWorkflowOutputs(ctx, tx, "run-outputs", test.slots)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if test.want == "" && verified != nil || test.want != "" && (verified == nil || verified.Error() != test.want) {
				t.Fatalf("verification = %v, want %q", verified, test.want)
			}
		})
	}
}
