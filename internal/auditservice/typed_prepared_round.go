package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func sealTypedSelectionRoot(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit) (auditstore.AdmissionMutation, error) {
	state := session.State
	root := auditdomain.RoundSelectionRoot{Schema: auditdomain.RoundSelectionRootSchema, AuditID: state.AuditID, RoundID: state.RoundID, Generation: state.Generation, ProfileDigest: state.ProfileDigest, BaselineDigest: state.BaselineDigest, PageCount: state.SelectionPages, ItemCount: state.SelectedCount, PagesDigest: state.SelectionPagesDigest}
	data, err := auditdomain.EncodeRoundSelectionRoot(root)
	if err != nil {
		return auditstore.AdmissionMutation{}, err
	}
	artifact, err := writeTypedAdmissionDocument(ctx, session, audit, "round-selection", auditdomain.RoundSelectionRootMediaType, data)
	return auditstore.AdmissionMutation{Next: state, SelectionRoot: &artifact}, err
}

func stageTypedPage(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit) (auditstore.AdmissionMutation, error) {
	state := session.State
	_, data, err := session.Reads.ReadSelectionPage(ctx, state.PreparedPages)
	if err != nil {
		return auditstore.AdmissionMutation{}, err
	}
	page, err := auditdomain.DecodeRoundSelectionPage(data)
	if err != nil {
		return auditstore.AdmissionMutation{}, err
	}
	if page.AuditID != state.AuditID || page.RoundID != state.RoundID || page.Generation != state.Generation || page.PageOrdinal != state.PreparedPages || page.FirstItemOrdinal != state.StagedCount {
		return auditstore.AdmissionMutation{}, inconsistentRound("a selected typed page differs from its durable generation", nil)
	}
	page.Schema = auditdomain.PreparedRoundPageSchema
	page.PreviousPageDigest = state.PagesDigest
	data, err = auditdomain.EncodePreparedRoundPage(page)
	if err != nil {
		return auditstore.AdmissionMutation{}, err
	}
	artifact, err := writeTypedAdmissionDocument(ctx, session, audit, "prepared-round-page", auditdomain.PreparedRoundPageMediaType, data)
	return auditstore.AdmissionMutation{Next: state, PreparedPage: &artifact, PageContent: data}, err
}

func sealTypedPreparedRoot(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit) (auditstore.AdmissionMutation, error) {
	state := session.State
	if state.SelectionRoot == nil {
		return auditstore.AdmissionMutation{}, inconsistentRound("typed prepared membership has no sealed selection source", nil)
	}
	root := auditdomain.PreparedRoundRoot{Schema: auditdomain.PreparedRoundRootSchema, AuditID: state.AuditID, RoundID: state.RoundID, Generation: state.Generation, ProfileDigest: state.ProfileDigest, BaselineDigest: state.BaselineDigest, Source: state.SelectionRoot.Clone(), PageCount: state.PreparedPages, ItemCount: state.StagedCount, PagesDigest: state.PagesDigest}
	data, err := auditdomain.EncodePreparedRoundRoot(root)
	if err != nil {
		return auditstore.AdmissionMutation{}, err
	}
	artifact, err := writeTypedAdmissionDocument(ctx, session, audit, "prepared-round", auditdomain.PreparedRoundRootMediaType, data)
	return auditstore.AdmissionMutation{Next: state, PreparedRoot: &artifact}, err
}

func writeTypedAdmissionDocument(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit, kind, mediaType string, data []byte) (auditstore.ExactArtifact, error) {
	writer := auditArtifactWriter{service: session.Artifacts, projectID: audit.ProjectID}
	return writer.write(ctx, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: typedAdmissionArtifactName(kind, session.State, auditdomain.DigestBytes(data))}, artifacts.Payload{MediaType: mediaType, Data: data})
}
