package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// HypothesisContent carries authored content only. Identity, revision, fixed
// membership and origin are assigned by the accepting owner.
type HypothesisContent struct {
	Key       string            `json:"key"`
	Statement string            `json:"statement"`
	Scope     map[string]string `json:"scope"`
	Grounds   string            `json:"grounds"`
}

type HypothesisMutationParams struct {
	MutationParams
	HypothesisID string
	Content      HypothesisContent
}

func (s *Service) WriteHypothesis(ctx context.Context, p HypothesisMutationParams) (auditstore.HypothesisAcceptance, bool, error) {
	if prior, found, err := s.owners().LookupHypothesisAcceptance(ctx, p.OwnerID, p.AuditID, p.IdempotencyKey, p.RequestDigest); err != nil || found {
		return prior, found, err
	}
	h := auditcontract.Hypothesis{Schema: auditcontract.HypothesisSchema, ID: auditdomain.DeterministicID("hypothesis", p.AuditID, p.IdempotencyKey, p.Content.Key), Revision: 1, Statement: p.Content.Statement, Scope: p.Content.Scope, Grounds: p.Content.Grounds, Origin: auditcontract.HypothesisOrigin{Kind: "manual", AuditID: p.AuditID}}
	source := p.Content.Key
	fixed := false
	if p.HypothesisID != "" {
		prior, err := s.reads().CurrentHypothesis(ctx, p.OwnerID, p.AuditID, p.HypothesisID)
		if err != nil {
			return auditstore.HypothesisAcceptance{}, false, err
		}
		if p.Content.Key != prior.SourceKey {
			return auditstore.HypothesisAcceptance{}, false, ErrInvalid
		}
		h.ID = prior.Document.ID
		h.Revision = prior.Document.Revision + 1
		h.Origin = prior.Document.Origin
		source = prior.SourceKey
		fixed = prior.Fixed
	}
	return s.owners().WriteHypotheses(ctx, auditstore.WriteHypothesesParams{OwnerID: p.OwnerID, AuditID: p.AuditID, ExpectedRevision: p.ExpectedRevision, IdempotencyKey: p.IdempotencyKey, RequestDigest: p.RequestDigest, Writes: []auditstore.HypothesisWrite{{Document: h, SourceKey: source, Fixed: fixed}}})
}

func (s *Service) Hypotheses(ctx context.Context, p auditstore.ListHypothesesParams) (auditstore.HypothesisPage, error) {
	return s.reads().ListHypotheses(ctx, p)
}
func (s *Service) Hypothesis(ctx context.Context, ownerID, auditID, id string, revision uint64) (auditstore.Hypothesis, error) {
	return s.reads().GetHypothesis(ctx, ownerID, auditID, id, revision)
}

func HypothesisRequestDigest(auditID, hypothesisID string, revision uint64, content HypothesisContent) (string, error) {
	return contentdigest.CanonicalJSON(map[string]any{"audit_id": auditID, "hypothesis_id": hypothesisID, "revision": revision, "content": content})
}
