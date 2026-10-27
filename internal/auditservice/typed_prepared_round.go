package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func sealTypedRoot(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit) (auditstore.AdmissionMutation, error) {
	state := session.State
	root := auditdomain.PreparedRoundRoot{Schema: auditdomain.PreparedRoundRootSchema, AuditID: state.AuditID, RoundID: state.RoundID, Generation: state.Generation, ProfileDigest: state.ProfileDigest, BaselineDigest: state.BaselineDigest, PageCount: state.PublishedPages, ItemCount: state.SelectedCount, PagesDigest: state.PagesDigest}
	data, err := auditdomain.EncodePreparedRoundRoot(root)
	if err != nil {
		return auditstore.AdmissionMutation{}, err
	}
	artifact, err := writeTypedAdmissionDocument(ctx, session, audit, "prepared-round", auditdomain.PreparedRoundRootMediaType, data)
	return auditstore.AdmissionMutation{Next: state, Root: &artifact}, err
}

func writeTypedAdmissionDocument(ctx context.Context, session *hookstore.AdmissionSession, audit auditstore.Audit, kind, mediaType string, data []byte) (auditstore.ExactArtifact, error) {
	writer := auditArtifactWriter{service: session.Artifacts, projectID: audit.ProjectID}
	return writer.write(ctx, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: typedAdmissionArtifactName(kind, session.State, auditdomain.DigestBytes(data))}, artifacts.Payload{MediaType: mediaType, Data: data})
}
