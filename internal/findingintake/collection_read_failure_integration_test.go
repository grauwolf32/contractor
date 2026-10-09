//go:build integration

package findingintake

import (
	"context"
	"encoding/hex"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestPostgresCollectionIsolatesProposalReadFailures(t *testing.T) {
	transient := errors.New("temporary blob storage outage")
	for _, scenario := range []struct {
		name, reason string
		err          error
	}{
		{"missing", "finding-proposal-artifact-missing", artifacts.ErrArtifactNotFound},
		{"corrupt", "finding-proposal-artifact-invalid", artifacts.ErrArtifactIntegrity},
		{"transient", "", transient},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			f := newDeletionImportFixture(t)
			bad := insertAuditChildReceiptWithEvidence(t, f, "bad", []ExactArtifact{})
			for _, key := range []string{"valid-a", "valid-b"} {
				insertAuditChildReceiptWithEvidence(t, f, key, []ExactArtifact{})
			}
			blobs := &collectionFaultBlobStore{
				digest: strings.TrimPrefix(bad.Proposal.Digest, "sha256:"), err: scenario.err,
			}
			f.ctx = artifacts.WithBlobRuntime(f.ctx, artifacts.NewBlobRuntime(blobs, nil))
			err := collectFixtureProposals(f.ctx, f, false)
			if scenario.reason == "" {
				if !errors.Is(err, transient) {
					t.Fatalf("transient read failure must remain retryable: %v", err)
				}
				assertFixtureHolds(t, f, 0)
			} else {
				if err != nil {
					t.Fatal(err)
				}
				assertFixtureHolds(t, f, 3)
			}
			var count int
			var reason string
			if err := f.pool.QueryRow(f.ctx, `
SELECT count(*), coalesce(min(summary ->> 'reason'), '') FROM audit_events
 WHERE audit_id = $1 AND kind = 'finding.proposal_rejected'`, f.request.AuditID).Scan(&count, &reason); err != nil {
				t.Fatal(err)
			}
			if reason != scenario.reason || count != boolCount(scenario.reason != "") {
				t.Fatalf("rejections: count=%d reason=%s", count, reason)
			}
			// Settled pages never reread the broken source or reject it twice.
			if scenario.reason != "" {
				reads := blobs.reads
				if err := collectFixtureProposals(f.ctx, f, false); err != nil || blobs.reads != reads {
					t.Fatalf("settled retry: reads=%d/%d error=%v", blobs.reads, reads, err)
				}
			}
		})
	}
}

type collectionFaultBlobStore struct {
	artifacts.PostgresBlobStore
	digest string
	err    error
	reads  int
}

func (s *collectionFaultBlobStore) Read(ctx context.Context, object artifacts.BlobObject) ([]byte, error) {
	s.reads++
	if hex.EncodeToString(object.Digest) == s.digest {
		return nil, s.err
	}
	return s.PostgresBlobStore.Read(ctx, object)
}

func boolCount(value bool) int {
	if value {
		return 1
	}
	return 0
}
