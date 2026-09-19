package auditservice

import (
	"errors"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestFindingPageValidatorsAllowOnlyOneCursorSentinel(t *testing.T) {
	validators := map[string]func(int) error{
		"findings": func(limit int) error {
			return validateFindingList(FindingListParams{OwnerID: "owner", AuditID: "audit", Limit: limit})
		},
		"reviews": func(limit int) error {
			return validateReviewList(ReviewListParams{OwnerID: "owner", AuditID: "audit", Limit: limit})
		},
		"provenance": func(limit int) error {
			return validateProvenanceList(ProvenanceListParams{OwnerID: "owner", AuditID: "audit", FindingID: "finding", Limit: limit})
		},
	}
	for name, validate := range validators {
		for _, limit := range []int{0, 1, 199, 200, 201, 202} {
			t.Run(fmt.Sprintf("%s/%d", name, limit), func(t *testing.T) {
				err := validate(limit)
				if limit >= 1 && limit <= 201 {
					if err != nil {
						t.Fatalf("bounded fetch rejected: %v", err)
					}
				} else if !errors.Is(err, auditstore.ErrInvalid) {
					t.Fatalf("unbounded fetch error = %v", err)
				}
			})
		}
	}
}
