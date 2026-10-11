//go:build integration

package auditservice

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// A serialization failure that outlasts every publication attempt stays a
// transaction conflict, which the public API reports as the retryable 503
// storage_transaction_conflict, not as a 409 resource conflict.
func TestFindingCollectionPublicationKeepsExhaustedTransactionConflicts(t *testing.T) {
	ctx, pool := collectionTestPool(t)
	seedCollectionRun(t, ctx, pool, "owner", "run-conflict")
	seedCollectionRunReceipt(t, ctx, pool, "owner", "run-conflict", "conflict", false)
	publisher := newCollectionTestPublisher(t, pool)
	if _, err := pool.Exec(ctx, `CREATE FUNCTION conflict_collection_receipt() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
IF NEW.namespace = 'finding-collection-receipts' THEN
    RAISE EXCEPTION 'injected serialization failure' USING ERRCODE = '`+postgres.SQLStateSerializationFailure+`';
END IF; RETURN NEW; END $$;
CREATE TRIGGER conflict_collection_receipt BEFORE INSERT ON artifact_bindings
FOR EACH ROW EXECUTE FUNCTION conflict_collection_receipt()`); err != nil {
		t.Fatal(err)
	}
	params := findingintake.PublishCollectionParams{OwnerID: "owner", Request: findingintake.PublishCollectionRequest{
		ClientKey: "conflicted", Sources: []findingintake.CollectionSelection{
			{Kind: "run", ID: "run-conflict", ReceiptIDs: []string{"receipt-conflict"}, Findings: []findingintake.CollectionFindingSelection{}},
		},
	}}
	_, err := publisher.PublishCollection(ctx, params)
	if !postgres.IsTransactionConflict(err) || errors.Is(err, findingintake.ErrConflict) {
		t.Fatalf("exhausted publication conflict = %v, want a transaction conflict", err)
	}
	user, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).User("owner")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := user.Read(ctx, contracts.ArtifactRef{Namespace: findingintake.CollectionNamespace, Name: "conflicted"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
		t.Fatalf("rolled-back collection is visible: %v", err)
	}
}
