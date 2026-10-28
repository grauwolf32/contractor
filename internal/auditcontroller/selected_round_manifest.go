package auditcontroller

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// PreparedRoundManifestLookup reads only the accepted immutable projections
// belonging to already selected item IDs. Its fitting prefix observes the
// caller's remaining metadata and related-descriptor budget.
type PreparedRoundManifestLookup interface {
	ReadSelectedPreparedManifest(context.Context, string, string, []string, int, int) ([]auditdomain.PreparedExecutionItem, error)
}

func (b *PinnedSubmissionBuilder) readSelectedRoundExecutionManifest(ctx context.Context, snapshot auditstore.ReconcileSnapshot, selected []CheckExecutionMember) (auditdomain.ExecutionManifest, bool, error) {
	audit, round := snapshot.Audit, snapshot.Round
	resolved, err := b.artifacts.ResolveProjectExact(ctx, audit.ProjectID, round.Manifest)
	if err != nil {
		return auditdomain.ExecutionManifest{}, false, err
	}
	if resolved.MediaType != auditdomain.PreparedRoundRootMediaType {
		manifest, err := b.readRoundExecutionManifest(ctx, audit.ProjectID, round.Manifest)
		return manifest, false, err
	}
	if resolved.SizeBytes > auditdomain.MaximumPreparedRoundRootBytes {
		return auditdomain.ExecutionManifest{}, true, invalidSubmission("prepared Round root exceeds its metadata limit")
	}
	payload, err := b.artifacts.ReadProjectExact(ctx, audit.ProjectID, resolved)
	if err != nil {
		return auditdomain.ExecutionManifest{}, true, err
	}
	root, err := auditdomain.DecodePreparedRoundRoot(payload.Data)
	if err != nil || root.AuditID != audit.AuditID || root.RoundID != round.RoundID || root.ProfileDigest != audit.Profile.Digest || root.BaselineDigest != auditdomain.DigestBytes(audit.BaselineSnapshot) || root.ItemCount != round.ExpectedItemCount || resolved.Ref.Namespace != auditdomain.ArtifactNamespace(audit.AuditID) {
		return auditdomain.ExecutionManifest{}, true, invalidSubmission("prepared Round root differs from accepted membership")
	}
	reader, ok := b.outputs.(PreparedRoundManifestLookup)
	if !ok {
		return auditdomain.ExecutionManifest{}, true, invalidSubmission("prepared Round selected-item reader is unavailable")
	}
	ids := make([]string, len(selected))
	for index, member := range selected {
		ids[index] = member.Item.ItemID
	}
	items, err := reader.ReadSelectedPreparedManifest(ctx, audit.AuditID, round.RoundID, ids, auditdomain.MaximumPreparedRoundPageBytes-len(payload.Data), auditdomain.MaximumPreparedRoundRelated-1)
	if err != nil {
		return auditdomain.ExecutionManifest{}, true, err
	}
	if len(items) == 0 || len(items) > len(selected) {
		return auditdomain.ExecutionManifest{}, true, invalidSubmission("prepared Round selected membership is empty or invalid")
	}
	manifest := auditdomain.ExecutionManifest{Schema: auditdomain.ExecutionManifestSchema, Items: make([]auditdomain.ExecutionItem, len(items))}
	local := auditdomain.ExecutionManifest{Schema: auditdomain.ExecutionManifestSchema, Items: make([]auditdomain.ExecutionItem, len(items))}
	for index, item := range items {
		selectedItem := selected[index].Item
		if auditdomain.ValidatePreparedExecutionItem(item) != nil || item.ItemKey != selectedItem.ItemKey || item.Ordinal != selectedItem.Ordinal || item.SubjectKey != selectedItem.SubjectKey {
			return auditdomain.ExecutionManifest{}, true, invalidSubmission("prepared Round projection differs from selected item")
		}
		manifest.Items[index] = item.ExecutionItem
		local.Items[index] = item.ExecutionItem
		local.Items[index].Ordinal = index
	}
	if auditdomain.ValidateDispatchExecutionManifest(local) != nil {
		return auditdomain.ExecutionManifest{}, true, invalidSubmission("prepared Round selected execution manifest is invalid")
	}
	return manifest, true, nil
}

func findSelectedPreparedManifestItem(manifest auditdomain.ExecutionManifest, item auditstore.Item) (auditdomain.ExecutionItem, error) {
	for _, candidate := range manifest.Items {
		if candidate.ItemKey == item.ItemKey && candidate.Ordinal == item.Ordinal && candidate.SubjectKey == item.SubjectKey {
			return candidate, nil
		}
	}
	return auditdomain.ExecutionItem{}, invalidSubmission("selected item is absent from its prepared Round projection")
}
