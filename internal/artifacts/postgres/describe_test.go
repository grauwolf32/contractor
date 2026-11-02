package postgres_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

// countingDB counts the statements a repository sends and whether any of
// them reads blob payloads.
type countingDB struct {
	persistencepostgres.DBTX
	mu           sync.Mutex
	statements   int
	payloadReads int
}

func (c *countingDB) record(sql string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.statements++
	if strings.Contains(sql, "payload") {
		c.payloadReads++
	}
}

func (c *countingDB) reset() (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	statements, payloadReads := c.statements, c.payloadReads
	c.statements, c.payloadReads = 0, 0
	return statements, payloadReads
}

func (c *countingDB) Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error) {
	c.record(sql)
	return c.DBTX.Exec(ctx, sql, args...)
}

func (c *countingDB) Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error) {
	c.record(sql)
	return c.DBTX.Query(ctx, sql, args...)
}

func (c *countingDB) QueryRow(ctx context.Context, sql string, args ...any) pgx.Row {
	c.record(sql)
	return c.DBTX.QueryRow(ctx, sql, args...)
}

// DescribeExact returns the descriptors Metadata reports for every exact
// revision in request order, reads them in one payload-free statement per
// MaxExactReadBatchSize references, and keeps an invalid reference distinct
// from a missing one and from a storage failure.
func TestDescribeExactReadsBoundedMetadataBatchesWithoutPayloads(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedArtifactPool(t, ctx)
	counter := &countingDB{DBTX: pool}
	service := NewService(artifactpgtest.NewStorage().Repository(counter))
	scope, _ := UserScope("describe-owner")
	store, err := service.Scoped(scope)
	if err != nil {
		t.Fatal(err)
	}
	const revisions = 3*MaxExactReadBatchSize + 5
	refs := make([]ArtifactRef, 0, revisions+1)
	want := make([]contracts.ExactArtifact, 0, revisions+1)
	logical := ArtifactRef{Namespace: "evidence", Name: "versions"}
	var previous *string
	for index := range revisions {
		written, err := store.Write(ctx, ArtifactRef{Namespace: "evidence", Name: fmt.Sprintf("item-%03d", index)},
			Payload{MediaType: "text/plain", Data: []byte(fmt.Sprintf("evidence %d", index))}, nil)
		if err != nil {
			t.Fatal(err)
		}
		refs = append(refs, written.Ref)
		if index < 2 {
			version, err := store.Write(ctx, logical,
				Payload{MediaType: "application/json", Data: []byte(fmt.Sprintf(`{"version":%d}`, index))}, previous)
			if err != nil {
				t.Fatal(err)
			}
			previous = version.Ref.Revision
			refs = append(refs, version.Ref)
		}
	}
	// A repeated reference is described at each position.
	refs = append(refs, refs[0])
	for _, ref := range refs {
		metadata, err := store.Metadata(ctx, ref)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, metadata.Exact())
	}

	counter.reset()
	described, err := store.DescribeExact(ctx, refs)
	if err != nil {
		t.Fatal(err)
	}
	statements, payloadReads := counter.reset()
	batches := (len(refs) + MaxExactReadBatchSize - 1) / MaxExactReadBatchSize
	if statements != batches || payloadReads != 0 {
		t.Fatalf("%d references used %d statements and %d payload reads, want %d and 0",
			len(refs), statements, payloadReads, batches)
	}
	if len(described) != len(want) {
		t.Fatalf("described %d references, want %d", len(described), len(want))
	}
	for index := range want {
		if want[index].Verify(described[index]) != nil || described[index].MediaType != want[index].MediaType ||
			described[index].SizeBytes != want[index].SizeBytes {
			t.Fatalf("descriptor %d = %+v, want %+v", index, described[index], want[index])
		}
	}

	missingRevision := "rev_missing"
	missing := ArtifactRef{Namespace: "evidence", Name: "item-000", Revision: &missingRevision}
	mixed := append(append([]ArtifactRef{}, refs[:MaxExactReadBatchSize+1]...), missing)
	if result, err := store.DescribeExact(ctx, mixed); result != nil || !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("missing revision = (%v, %v), want ErrArtifactNotFound", result, err)
	}
	foreignScope, _ := UserScope("another-owner")
	foreign, err := service.Scoped(foreignScope)
	if err != nil {
		t.Fatal(err)
	}
	if result, err := foreign.DescribeExact(ctx, refs[:1]); result != nil || !errors.Is(err, ErrArtifactNotFound) {
		t.Fatalf("foreign-scope revision = (%v, %v), want ErrArtifactNotFound", result, err)
	}
	counter.reset()
	for name, invalid := range map[string]struct {
		ref  ArtifactRef
		want error
	}{
		"latest alias": {ArtifactRef{Namespace: "evidence", Name: "item-000"}, ErrExactRevisionRequired},
		"invalid name": {ArtifactRef{Namespace: "evidence", Name: "../escape", Revision: &missingRevision}, ErrInvalidName},
	} {
		request := append(append([]ArtifactRef{}, refs...), invalid.ref)
		if result, err := store.DescribeExact(ctx, request); result != nil || !errors.Is(err, invalid.want) {
			t.Fatalf("%s = (%v, %v), want %v", name, result, err, invalid.want)
		}
	}
	if statements, _ := counter.reset(); statements != 0 {
		t.Fatalf("invalid references sent %d statements, want none", statements)
	}
	if _, err := artifactpgtest.NewStorage().Repository(pool).DescribeExact(ctx, scope, make([]ArtifactRef, MaxExactReadBatchSize+1)); !errors.Is(err, blob.ErrTooLarge) {
		t.Fatalf("oversized repository batch = %v, want ErrPayloadTooLarge", err)
	}
	cancelled, cancelNow := context.WithCancel(ctx)
	cancelNow()
	_, err = store.DescribeExact(cancelled, refs)
	if err == nil || errors.Is(err, ErrArtifactNotFound) || errors.Is(err, ErrExactRevisionRequired) ||
		errors.Is(err, ErrInvalidName) {
		t.Fatalf("cancelled description = %v, want a storage failure", err)
	}
}
