package auditservice

import (
	"context"
	"strings"
	"unicode/utf8"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/reviewstore"
	"github.com/jackc/pgx/v5"
)

func (s *Service) GetReview(
	ctx context.Context, ownerID, auditID, requestID string,
) (auditstore.ReviewRequest, error) {
	return s.reads().GetReview(ctx, ownerID, auditID, requestID)
}

func (s *Service) ListReviews(
	ctx context.Context, params auditstore.ReviewListParams,
) ([]auditstore.ReviewRequest, error) {
	return s.reads().ListReviews(ctx, params)
}

func appendAuditReviewEvent(
	ctx context.Context, tx pgx.Tx, auditID, kind, entityID string, summary map[string]any,
	storage *artifactpostgres.Storage,
) error {
	return reviewstore.New(tx, storage).AppendReviewEvent(ctx, auditstore.ReviewEventParams{
		AuditID: auditID, Kind: kind, EntityID: entityID, Summary: summary,
	})
}

func validRationale(value string) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= MaxReviewRationale
}
