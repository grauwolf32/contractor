package projectstore

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// CreateRequestDigest is the canonical digest of a Project creation request
// that its Idempotency-Key binds (API-R69, PER-R24): the kind, name,
// description and active-checks policy, an omitted policy bound as automatic
// as Create stores it. Create replays a repeated key with the same digest and
// reports the idempotency conflict for another one.
func CreateRequestDigest(params CreateParams) (string, error) {
	activeChecks := params.ActiveChecks
	if activeChecks == "" {
		activeChecks = contracts.ActiveChecksAutomatic
	}
	digest, err := contentdigest.JSON(struct {
		Kind         Kind                         `json:"kind"`
		Name         string                       `json:"name"`
		Description  string                       `json:"description"`
		ActiveChecks contracts.ActiveChecksPolicy `json:"activeChecks"`
	}{params.Kind, params.Name, params.Description, activeChecks})
	if err != nil {
		return "", fmt.Errorf("encode Project request: %w", err)
	}
	return digest, nil
}
