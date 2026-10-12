package auditpriority

import (
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// Verdict is one model's independent assessment of its assigned checklist item.
// Evidence IDs refer to context facts, not accepted vulnerability evidence.
// Both slices are required on the wire; empty arrays are valid, null is not.
type Verdict struct {
	ItemKey        string     `json:"item_key"`
	Priority       Priority   `json:"priority"`
	Confidence     Confidence `json:"confidence"`
	Rationale      string     `json:"rationale"`
	EvidenceIDs    []string   `json:"evidence_ids"`
	MissingContext []string   `json:"missing_context"`
}

// verdictBounds admit one object of six members, two of them arrays of at
// most MaxEvidenceIDs and MaxMissingContextEntries strings; nothing nests
// below an array element.
var verdictBounds = strictjson.Bounds{MaxDepth: 3, MaxNodes: 1 + 6 + MaxEvidenceIDs + MaxMissingContextEntries}

// DecodeVerdict accepts exactly one bounded object with the six required,
// case-sensitive fields: four strings and two string arrays, and no number.
// It checks assignment and evidence authority against the caller's frozen
// context. All failures return a closed diagnostic code.
func DecodeVerdict(data []byte, expectedItemKey string, contextEvidenceIDs []string) (Verdict, error) {
	bad := func() (Verdict, error) { return Verdict{}, invalid(CodeInvalidVerdict) }
	if len(data) == 0 || len(data) > MaxVerdictBytes || !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) {
		return bad()
	}
	document, err := strictjson.DecodeBounded(data, verdictBounds, func(string) (struct{}, bool) {
		return struct{}{}, false
	})
	fields, isObject := document.(map[string]any)
	if err != nil || !isObject || len(fields) != 6 {
		return bad()
	}
	itemKey, itemKeyOK := fields["item_key"].(string)
	priority, priorityOK := fields["priority"].(string)
	confidence, confidenceOK := fields["confidence"].(string)
	rationale, rationaleOK := fields["rationale"].(string)
	evidenceIDs, evidenceOK := verdictStrings(fields["evidence_ids"], MaxEvidenceIDs)
	missingContext, missingOK := verdictStrings(fields["missing_context"], MaxMissingContextEntries)
	if !itemKeyOK || !priorityOK || !confidenceOK || !rationaleOK || !evidenceOK || !missingOK {
		return bad()
	}
	verdict := Verdict{
		ItemKey: itemKey, Priority: Priority(priority), Confidence: Confidence(confidence),
		Rationale: rationale, EvidenceIDs: evidenceIDs, MissingContext: missingContext,
	}
	if err := ValidateVerdict(verdict, expectedItemKey, contextEvidenceIDs); err != nil {
		return bad()
	}
	return verdict, nil
}

// ValidateVerdict checks structure, encoded size, exact assigned item identity
// and membership in the caller's bounded, unique context evidence IDs.
func ValidateVerdict(verdict Verdict, expectedItemKey string, contextEvidenceIDs []string) error {
	if !validIdentifier(expectedItemKey) || verdict.ItemKey != expectedItemKey || !validEvidenceIDs(contextEvidenceIDs) {
		return invalid(CodeInvalidVerdict)
	}
	if _, err := MarshalVerdict(verdict); err != nil {
		return err
	}
	allowed := make(map[string]bool, len(contextEvidenceIDs))
	for _, id := range contextEvidenceIDs {
		allowed[id] = true
	}
	for _, id := range verdict.EvidenceIDs {
		if !allowed[id] {
			return invalid(CodeInvalidVerdict)
		}
	}
	return nil
}

// MarshalVerdict validates structural bounds and returns RFC 8785 canonical
// bytes. It cannot establish assignment or context authority: callers must use
// ValidateVerdict with the exact assigned item and frozen context evidence IDs.
func MarshalVerdict(verdict Verdict) ([]byte, error) {
	if !validIdentifier(verdict.ItemKey) || !validText(verdict.Rationale, MaxRationaleBytes) ||
		verdict.EvidenceIDs == nil || len(verdict.EvidenceIDs) > MaxEvidenceIDs || !validEvidenceIDs(verdict.EvidenceIDs) ||
		verdict.MissingContext == nil || len(verdict.MissingContext) > MaxMissingContextEntries {
		return nil, invalid(CodeInvalidVerdict)
	}
	switch verdict.Priority {
	case PriorityCritical, PriorityHigh, PriorityMedium, PriorityLow:
	default:
		return nil, invalid(CodeInvalidVerdict)
	}
	switch verdict.Confidence {
	case ConfidenceHigh, ConfidenceMedium, ConfidenceLow:
	default:
		return nil, invalid(CodeInvalidVerdict)
	}
	seen := make(map[string]bool, len(verdict.MissingContext))
	for _, missing := range verdict.MissingContext {
		if !validText(missing, MaxMissingContextBytes) || seen[missing] {
			return nil, invalid(CodeInvalidVerdict)
		}
		seen[missing] = true
	}
	data, err := contracts.MarshalPrivateCanonical(verdict)
	if err != nil || len(data) > MaxVerdictBytes {
		return nil, invalid(CodeInvalidVerdict)
	}
	return data, nil
}

func cloneVerdict(verdict Verdict) Verdict {
	if verdict.EvidenceIDs != nil {
		verdict.EvidenceIDs = append([]string{}, verdict.EvidenceIDs...)
	}
	if verdict.MissingContext != nil {
		verdict.MissingContext = append([]string{}, verdict.MissingContext...)
	}
	return verdict
}

func validEvidenceIDs(ids []string) bool {
	if len(ids) > MaxContextEvidence {
		return false
	}
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if !validIdentifier(id) || seen[id] {
			return false
		}
		seen[id] = true
	}
	return true
}

// verdictStrings reads one verdict string array of at most limit entries; an
// empty array is an empty, non-nil slice.
func verdictStrings(value any, limit int) ([]string, bool) {
	elements, ok := value.([]any)
	if !ok || len(elements) > limit {
		return nil, false
	}
	result := make([]string, len(elements))
	for index, element := range elements {
		if result[index], ok = element.(string); !ok {
			return nil, false
		}
	}
	return result, true
}
