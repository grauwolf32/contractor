package auditcontract

import (
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"unicode/utf8"
)

const HypothesisSchema = "contractor.audit.hypothesis.v1"

// Hypothesis is an exact assertion with identity and provenance assigned by its
// accepting owner. Editing content never changes an earlier revision.
type Hypothesis struct {
	Schema    string            `json:"schema"`
	ID        string            `json:"id"`
	Revision  uint64            `json:"revision"`
	Statement string            `json:"statement"`
	Scope     map[string]string `json:"scope"`
	Grounds   string            `json:"grounds"`
	Origin    HypothesisOrigin  `json:"origin"`
}

type HypothesisOrigin struct {
	Kind    string    `json:"kind"`
	AuditID string    `json:"audit_id"`
	TaskID  string    `json:"task_id,omitempty"`
	RunID   string    `json:"run_id,omitempty"`
	Source  *Material `json:"source,omitempty"`
}

func DecodeHypothesis(data []byte) (Hypothesis, error) {
	canonical, err := Decode(data, nil)
	if err != nil {
		return Hypothesis{}, err
	}
	var h Hypothesis
	if strictjson.Decode(canonical, &h) != nil || h.Schema != HypothesisSchema {
		return Hypothesis{}, errInvalid
	}
	return h, nil
}

// DecodeHypothesisSeed validates content before identity assignment against the
// same seed schema and canonical byte bound used by programs and proposals.
func DecodeHypothesisSeed(data []byte) (HypothesisSeed, error) {
	var value any
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) || strictjson.Decode(data, &value) != nil || !validBytes(value) {
		return HypothesisSeed{}, errInvalid
	}
	compiled, err := seedSchema()
	if err != nil {
		return HypothesisSeed{}, err
	}
	if compiled.Validate(value) != nil {
		return HypothesisSeed{}, errInvalid
	}
	canonical, err := strictjson.Canonical(value)
	if err != nil || len(canonical) > documentLimit(HypothesisSchema) {
		return HypothesisSeed{}, errInvalid
	}
	var seed HypothesisSeed
	if strictjson.Decode(canonical, &seed) != nil {
		return HypothesisSeed{}, errInvalid
	}
	return seed, nil
}
