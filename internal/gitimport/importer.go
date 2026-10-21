package gitimport

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/crypto/ssh"
)

var (
	ErrCapacity = apierror.New(apierror.GitImportCapacity, "Git import capacity is exhausted")
	// ErrTimedOut and ErrInterrupted report an import that ended with its
	// context: the whole-import deadline expired, or the request was
	// cancelled. Nothing partial is published (GIT-R73).
	ErrTimedOut    = apierror.New(apierror.GitImportTimeout, "Git import deadline expired")
	ErrInterrupted = apierror.New(apierror.GitImportCancelled, "Git import was interrupted")
)

type SnapshotClient interface {
	Allowed(Remote) bool
	Fetch(context.Context, Remote, string, Mode, ssh.Signer) (Snapshot, error)
}
type KeyProvider interface {
	Signer(context.Context, string) (ssh.Signer, error)
}
type ImportRequest struct {
	Mode             Mode
	OwnerID          string
	ProjectID        string
	Target           artifacts.ArtifactRef
	ExpectedRevision *string
	RepositoryURL    string
	Ref              string
}
type ImportResult struct {
	Artifact  artifacts.ArtifactRef `json:"artifact"`
	MediaType string                `json:"mediaType"`
	Size      int64                 `json:"size"`
	GitSource artifacts.GitSource   `json:"gitSource"`
}
type Importer struct {
	pool    *pgxpool.Pool
	client  SnapshotClient
	keys    KeyProvider
	slots   chan struct{}
	timeout time.Duration
}

func NewImporter(pool *pgxpool.Pool, client SnapshotClient, keys KeyProvider, timeout time.Duration) (*Importer, error) {
	if pool == nil || client == nil || keys == nil || timeout <= 0 || timeout > 30*time.Minute {
		return nil, ErrConfiguration
	}
	return &Importer{pool: pool, client: client, keys: keys, slots: make(chan struct{}, 1), timeout: timeout}, nil
}

func (i *Importer) Timeout() time.Duration { return i.timeout }

// DoImport keeps both admission leases through the caller's small metadata
// response. The complete ZIP is never passed to the HTTP response callback.
func (i *Importer) DoImport(ctx context.Context, request ImportRequest, respond func(ImportResult)) (err error) {
	defer func() { err = importEnded(err) }()
	if request.Mode == "" {
		request.Mode = ModeSnapshot
	}
	if request.Mode != ModeSnapshot && request.Mode != ModeAnalysis {
		return ErrContent
	}
	if respond == nil {
		return ErrConfiguration
	}
	ctx, cancel := context.WithTimeout(ctx, i.timeout)
	defer cancel()
	ctx, release, err := artifacts.AcquireTransfer(ctx)
	if err != nil {
		return err
	}
	defer release()
	select {
	case i.slots <- struct{}{}:
	default:
		return ErrCapacity
	}
	defer func() { <-i.slots }()
	if request.Target.Revision != nil {
		return artifacts.ErrVersionedWriteTarget
	}
	if (request.ProjectID == "" && artifactpolicy.IsAuditStandardCatalogNamespace(request.Target.Namespace)) || (request.ProjectID != "" && artifactpolicy.IsAuditManagedProjectNamespace(request.Target.Namespace)) {
		return artifacts.ErrReservedNamespace
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(i.pool))
	store, scope, err := importScope(service, request)
	if err != nil {
		return err
	}
	if request.ProjectID != "" {
		project, err := projectstore.NewPostgresStore(i.pool).Get(ctx, request.OwnerID, request.ProjectID)
		if err != nil {
			return err
		}
		if project.Lifecycle == projectstore.LifecycleDeleting {
			return projectstore.ErrDeleting
		}
	}
	if err := checkImportPrecondition(ctx, store, request.Target, request.ExpectedRevision); err != nil {
		return err
	}
	remote, err := ParseRemote(request.RepositoryURL)
	if err != nil {
		return err
	}
	if !i.client.Allowed(remote) {
		return ErrDestination
	}
	var signer ssh.Signer
	if remote.Scheme == "ssh" {
		signer, err = i.keys.Signer(ctx, request.OwnerID)
		if err != nil {
			return err
		}
		if signer == nil {
			return credentials.ErrGitKeyMissing
		}
	}
	// Every query above has returned its connection before remote work begins.
	snapshot, err := i.client.Fetch(ctx, remote, request.Ref, request.Mode, signer)
	if err != nil {
		return err
	}
	if (request.Mode == ModeAnalysis) != (snapshot.Analysis != nil) {
		return ErrContent
	}
	payload, err := artifacts.PreparePayload(ctx, artifacts.Payload{MediaType: "application/zip", Data: snapshot.Data})
	if err != nil {
		return err
	}
	source := artifacts.GitSource{Analysis: snapshot.Analysis, RepositoryURL: snapshot.RepositoryURL, RequestedRef: snapshot.RequestedRef, ResolvedCommit: snapshot.Commit, ImportedAt: time.Now().UTC().Truncate(time.Microsecond)}
	var result artifacts.WriteResult
	err = postgres.InTx(ctx, i.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if request.ProjectID != "" {
			var state string
			err := tx.QueryRow(ctx, `SELECT lifecycle_state FROM projects WHERE owner_id=$1 AND project_id=$2 FOR SHARE`, request.OwnerID, request.ProjectID).Scan(&state)
			if errors.Is(err, pgx.ErrNoRows) {
				return projectstore.ErrNotFound
			}
			if err != nil {
				return err
			}
			if state != "active" {
				return projectstore.ErrDeleting
			}
		}
		repository := artifacts.NewPostgresTxRepository(tx)
		store, _, err := importScope(artifacts.NewService(repository), request)
		if err != nil {
			return err
		}
		result, err = store.Write(ctx, request.Target, payload, request.ExpectedRevision)
		if err != nil {
			return err
		}
		return repository.RecordGitSource(ctx, scope, result.Ref, source)
	})
	if err != nil {
		return err
	}
	respond(ImportResult{Artifact: result.Ref, MediaType: result.MediaType, Size: result.Size, GitSource: source})
	return nil
}

// importEnded classifies an import that ended because its context did as a
// timeout or an interruption, unless one of the import's own errors already
// explains it.
func importEnded(err error) error {
	var unsupported *UnsupportedEntryError
	switch {
	case err == nil, errors.As(err, &unsupported):
		return err
	}
	for _, own := range []error{
		ErrURL, ErrRef, ErrDestination, ErrTrust, ErrRemote, ErrContent, ErrBudget, ErrCapacity, ErrConfiguration,
	} {
		if errors.Is(err, own) {
			return err
		}
	}
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return fmt.Errorf("%w: %w", ErrTimedOut, err)
	case errors.Is(err, context.Canceled):
		return fmt.Errorf("%w: %w", ErrInterrupted, err)
	}
	return err
}

func importScope(service *artifacts.Service, request ImportRequest) (artifacts.ScopedStore, artifacts.Scope, error) {
	if request.ProjectID != "" {
		store, err := service.Project(request.ProjectID)
		scope, _ := artifacts.ProjectScope(request.ProjectID)
		return store, scope, err
	}
	store, err := service.User(request.OwnerID)
	scope, _ := artifacts.UserScope(request.OwnerID)
	return store, scope, err
}
func checkImportPrecondition(ctx context.Context, store artifacts.ScopedStore, target artifacts.ArtifactRef, expected *string) error {
	metadata, err := store.Metadata(ctx, target)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return artifacts.WritePrecondition(target, nil, expected)
	}
	if err != nil {
		return err
	}
	return artifacts.WritePrecondition(target, &metadata, expected)
}
