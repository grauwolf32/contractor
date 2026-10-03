package auditpriority

import (
	"bytes"
	"encoding/json"
	"io"
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

// DecodeVerdict accepts exactly one bounded object with the six required,
// case-sensitive fields. It checks assignment and evidence authority against
// the caller's frozen context. All failures return a closed diagnostic code.
func DecodeVerdict(data []byte, expectedItemKey string, contextEvidenceIDs []string) (Verdict, error) {
	var verdict Verdict
	bad := func() (Verdict, error) { return Verdict{}, invalid(CodeInvalidVerdict) }
	if len(data) == 0 || len(data) > MaxVerdictBytes || !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) {
		return bad()
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return bad()
	}
	seen := make(map[string]bool, 6)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return bad()
		}
		key, ok := token.(string)
		if !ok || seen[key] {
			return bad()
		}
		seen[key] = true
		switch key {
		case "item_key", "priority", "confidence", "rationale":
			token, err = decoder.Token()
			value, isString := token.(string)
			if err != nil || !isString {
				return bad()
			}
			switch key {
			case "item_key":
				verdict.ItemKey = value
			case "priority":
				verdict.Priority = Priority(value)
			case "confidence":
				verdict.Confidence = Confidence(value)
			case "rationale":
				verdict.Rationale = value
			}
		case "evidence_ids":
			verdict.EvidenceIDs, err = decodeVerdictStrings(decoder, MaxEvidenceIDs)
			if err != nil {
				return bad()
			}
		case "missing_context":
			verdict.MissingContext, err = decodeVerdictStrings(decoder, MaxMissingContextEntries)
			if err != nil {
				return bad()
			}
		default:
			return bad()
		}
	}
	if len(seen) != 6 {
		return bad()
	}
	if token, err := decoder.Token(); err != nil || token != json.Delim('}') {
		return bad()
	}
	if _, err := decoder.Token(); err != io.EOF {
		return bad()
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

// The decoder accepts only strings at depth one and string arrays at depth
// two. Nested objects/arrays are rejected immediately; there is no recursive
// walk or unbounded allocation for adversarial JSON structure.
func decodeVerdictStrings(decoder *json.Decoder, limit int) ([]string, error) {
	if token, err := decoder.Token(); err != nil || token != json.Delim('[') {
		return nil, invalid(CodeInvalidVerdict)
	}
	values := make([]string, 0)
	for decoder.More() {
		if len(values) == limit {
			return nil, invalid(CodeInvalidVerdict)
		}
		token, err := decoder.Token()
		value, ok := token.(string)
		if err != nil || !ok {
			return nil, invalid(CodeInvalidVerdict)
		}
		values = append(values, value)
	}
	if token, err := decoder.Token(); err != nil || token != json.Delim(']') {
		return nil, invalid(CodeInvalidVerdict)
	}
	return values, nil
}
