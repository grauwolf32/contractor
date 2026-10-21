//go:build integration

package auditcontroller

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/rundeletion"
)

func TestPostgresCompositionReadsOnlyCommittedInventoryAndRetainedOutputs(t *testing.T) {
	ctx, h, c := generatedInventoryHarness(t)
	readyGeneratedInventory(t, ctx, h, c)
	claim, snapshot := initialInventoryClaim(t, ctx, h)
	owner, id := snapshot.Audit.OwnerID, snapshot.Audit.AuditID
	params, reason, err := h.service.PrepareInitialRound(ctx, claim, snapshot)
	if err != nil || reason != nil {
		t.Fatalf("prepare: %v %v", reason, err)
	}
	if view, err := h.service.GeneratedInventory(ctx, owner, id, snapshot.Audit.Revision); err != nil || view != nil {
		t.Fatalf("staging became inventory: %+v %v", view, err)
	}
	ref := params.Inventory.Artifact.Ref
	if _, err := h.service.ReadArtifact(ctx, owner, id, auditstore.InitialInventoryLogicalKey, *ref.Revision); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("staging became readable: %v", err)
	}
	if _, _, err := h.audits.AcceptInitialRound(ctx, params); err != nil {
		t.Fatal(err)
	}
	audit, err := h.audits.Get(ctx, owner, id)
	if err != nil {
		t.Fatal(err)
	}
	view, err := h.service.GeneratedInventory(ctx, owner, id, audit.Revision)
	if err != nil || view == nil || view.ItemCount != len(params.Items) || !view.Snapshot.Ref.SameExact(ref) || !view.Worklist.Ref.SameExact(params.Manifest.Ref) {
		t.Fatalf("accepted inventory: %+v %v", view, err)
	}
	if _, err := h.service.GeneratedInventory(ctx, owner, id, snapshot.Audit.Revision); !errors.Is(err, auditstore.ErrConflict) {
		t.Fatalf("stale context: %v", err)
	}
	if _, err := h.service.GeneratedInventory(ctx, "foreign-owner", id, audit.Revision); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign inventory: %v", err)
	}
	prepared := preparationExecutions(t, ctx, h, 1)[0]
	accepted, err := h.audits.GetPreparationOutput(ctx, id, "z-seed", "list")
	if err != nil {
		t.Fatal(err)
	}
	key, revision := "prepare:"+prepared.ExecutionID+":list", *accepted.Output.Retained.Ref.Revision
	if _, err := h.service.ReadArtifact(ctx, owner, id, key, revision); err != nil {
		t.Fatalf("accepted output: %v", err)
	}
	for _, request := range [][3]string{{"foreign-owner", key, revision}, {owner, "uncommitted", revision}, {owner, key, "other-revision"}} {
		if _, err := h.service.ReadArtifact(ctx, request[0], id, request[1], request[2]); !errors.Is(err, auditstore.ErrNotFound) {
			t.Fatalf("unauthorized or mutable lookup: %v", err)
		}
	}
	if err := rundeletion.New(h.pool).DeleteReleasedTerminalRun(ctx, owner, *prepared.RunID); err != nil {
		t.Fatal(err)
	}
	if _, err := h.service.ReadArtifact(ctx, owner, id, key, revision); err != nil {
		t.Fatalf("source Run deletion removed retained output: %v", err)
	}
	audit, _ = h.audits.Get(ctx, owner, id)
	if retained, err := h.service.GeneratedInventory(ctx, owner, id, audit.Revision); err != nil || retained == nil || retained.Sources["source"].Digest != accepted.Output.Retained.Digest {
		t.Fatalf("source deletion changed inventory provenance: %+v %v", retained, err)
	}
}
