package projectlifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// On the filesystem blob backend, the purge of a Project's Audit and the
// final purge of the Project unlink the object of every blob they delete and
// keep each object a surviving binding still references (BLB-R50, BLB-R51):
// no purge leaves an orphan file or a missing reference behind. The test
// always claims the filesystem backend, so every database pass runs it.
func TestProjectAndAuditPurgesUnlinkOnlyUnreferencedFilesystemBlobs(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPool(t, ctx)
	if err := artifacts.ClaimBlobBackend(ctx, pool, artifacts.BlobFilesystem); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	files, err := artifacts.OpenFilesystemBlobStore(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	defer files.Close()
	ctx = artifacts.WithBlobRuntime(ctx, artifacts.NewBlobRuntime(files, nil))

	projects := projectstore.NewPostgresStore(pool)
	runs := runstore.NewPostgresStore(pool)
	audits := auditstore.NewPostgresStore(pool)
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-files", OwnerID: "user-1", Kind: projectstore.KindProject,
		Name: "Filesystem purge", IdempotencyKey: "create-project-files",
		RequestDigest: "sha256:" + strings.Repeat("a", 64),
	})
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: "audit-files", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:         auditstore.ProfileIdentity{Name: "checklist", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64)},
		ProfileSnapshot: json.RawMessage(`{"ref":{"name":"checklist","version":"1"}}`),
		InputSelection:  json.RawMessage(`{}`),
		Limits: auditstore.Limits{
			MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024,
		},
		IdempotencyKey: "create-audit-files", RequestDigest: "sha256:" + strings.Repeat("c", 64),
	})
	if err != nil {
		t.Fatal(err)
	}

	// Each content is one blob; a blob is shared by every binding of its bytes.
	const (
		shared       = "bytes of the owner and the Project"
		projectOnly  = "bytes of the Project alone"
		auditOnly    = "bytes of the Audit alone"
		auditProject = "bytes of the Audit and the Project"
		runOnly      = "bytes of the Project Run alone"
	)
	write := func(store func(string) (artifacts.ScopedStore, error), scopeID, namespace, name, content string) {
		t.Helper()
		scoped, err := store(scopeID)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := scoped.Write(ctx, artifacts.ArtifactRef{Namespace: namespace, Name: name},
			artifacts.Payload{MediaType: "text/plain", Data: []byte(content)}, nil); err != nil {
			t.Fatal(err)
		}
	}
	write(service.User, project.OwnerID, "sources", "shared", shared)
	write(service.Project, project.ProjectID, "sources", "shared", shared)
	write(service.Project, project.ProjectID, "sources", "project-only", projectOnly)
	write(service.Project, project.ProjectID, "sources", "audit-copy", auditProject)
	namespace := auditdomain.ArtifactNamespace(audit.AuditID)
	for name, content := range map[string]string{"report": auditOnly, "copy": auditProject} {
		if _, err := service.WriteAuditArtifact(ctx, project.ProjectID,
			artifacts.ArtifactRef{Namespace: namespace, Name: name},
			artifacts.Payload{MediaType: "text/plain", Data: []byte(content)}); err != nil {
			t.Fatal(err)
		}
	}
	run := createProjectRun(t, ctx, runs, "run-files", project.ProjectID)
	write(service.Run, run.RunID, "scratch", "run-only", runOnly)
	if _, err := runs.TransitionRun(
		ctx, run.RunID, runstore.RunInitializing, runstore.RunFailed, runstore.Reason{Code: "fixture_failed"},
	); err != nil {
		t.Fatal(err)
	}

	objects := map[string]string{}
	for _, content := range []string{shared, projectOnly, auditOnly, auditProject, runOnly} {
		digest := sha256.Sum256([]byte(content))
		var key string
		if err := pool.QueryRow(ctx, `
SELECT object_key FROM artifact_blobs WHERE backend = 'filesystem' AND sha256 = $1`, digest[:]).Scan(&key); err != nil {
			t.Fatalf("blob of %q: %v", content, err)
		}
		objects[content] = key
	}
	assertObjects := func(stage string, present, absent []string, referenced int64) {
		t.Helper()
		for _, content := range present {
			if _, err := os.Stat(filepath.Join(root, objects[content])); err != nil {
				t.Errorf("%s: object of %q: %v, want it kept", stage, content, err)
			}
		}
		for _, content := range absent {
			if _, err := os.Stat(filepath.Join(root, objects[content])); !errors.Is(err, os.ErrNotExist) {
				t.Errorf("%s: object of %q: %v, want it unlinked", stage, content, err)
			}
		}
		report, err := artifacts.CleanupFilesystemBlobs(ctx, pool, root, false)
		if err != nil || report.Referenced != referenced || report.Orphans != 0 || report.Missing != 0 {
			t.Fatalf("%s: filesystem blobs = (%+v, %v), want %d referenced, no orphan or missing object",
				stage, report, err, referenced)
		}
	}
	assertObjects("before deletion", []string{shared, projectOnly, auditOnly, auditProject, runOnly}, nil, 5)

	deleting, accepted, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
	})
	if err != nil || !accepted || deleting.Lifecycle != projectstore.LifecycleDeleting {
		t.Fatalf("begin deletion = (%+v, %t, %v)", deleting, accepted, err)
	}
	controller := newTestController(t, pool, runs, &recordingNotifier{}, "files")
	// The controller asks the Audit to delete itself, then waits for its purge.
	for iteration := 0; ; iteration++ {
		worked, err := controller.RunOnce(ctx)
		if err != nil {
			t.Fatalf("iteration %d before the Audit purge: %v", iteration, err)
		}
		if !worked {
			break
		}
		if iteration == 8 {
			t.Fatal("the Project deletion never waited for its Audit")
		}
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: "files-audit-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 || claims[0].AuditID != audit.AuditID {
		t.Fatalf("claim deleting Audit = (%+v, %v)", claims, err)
	}
	if err := audits.PurgeClaimed(ctx, claims[0], namespace); err != nil {
		t.Fatalf("purge deleting Audit: %v", err)
	}
	assertObjects("after the Audit purge", []string{shared, projectOnly, auditProject, runOnly}, []string{auditOnly}, 4)

	expireDeletionClaim(t, ctx, pool, project.ProjectID)
	for iteration := 0; ; iteration++ {
		if _, err := controller.RunOnce(ctx); err != nil {
			t.Fatalf("iteration %d after the Audit purge: %v", iteration, err)
		}
		if _, err := projects.Get(ctx, project.OwnerID, project.ProjectID); errors.Is(err, projectstore.ErrNotFound) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if iteration == 12 {
			t.Fatal("the Project deletion never finished")
		}
	}
	assertObjects("after the Project purge", []string{shared}, []string{projectOnly, auditOnly, auditProject, runOnly}, 1)
}
