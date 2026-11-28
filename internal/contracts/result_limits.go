package contracts

import (
	"bytes"
	"encoding/json"
)

// The StageContentRequest bound of the private wire. The Stage result bounds
// belong to the Planner (stageresult.MaxStageResultBytes); keep their
// identities separate even when their current numeric values coincide.
const (
	MaxStageRequestBytes = 256 * 1024
	// This bounds a WorkerResult payload at the Planner boundary; the complete
	// WorkerCompletion envelope has its own MaxWorkerCompletionBytes limit.
	MaxWorkerResultPayloadBytes = 256 * 1024
)

// ResultJSONSize counts compact UTF-8 JSON bytes, as the Runtime result models
// do. HTML and JavaScript-only escaping must not reduce the shared result budget.
// Size accounting retains integer digits and does not decode through float64.
func ResultJSONSize(value any) (int, error) {
	encoded, err := EncodeCompactJSON(value)
	return len(encoded), err
}

// EncodeCompactJSON preserves exact numbers and emits the same compact UTF-8
// representation counted by ResultJSONSize, without HTML/JavaScript escaping.
// It is not a canonical digest encoding.
func EncodeCompactJSON(value any) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return nil, err
	}
	encoded := buffer.Bytes()[:buffer.Len()-1] // Encoder's trailing newline.
	result := make([]byte, 0, len(encoded))
	for i := 0; i+1 < len(encoded); i++ {
		if encoded[i] != '\\' {
			continue
		}
		if i+6 <= len(encoded) && (string(encoded[i:i+6]) == `\u2028` || string(encoded[i:i+6]) == `\u2029`) {
			result = append(result, encoded[:i]...)
			if encoded[i+5] == '8' {
				result = append(result, "\u2028"...)
			} else {
				result = append(result, "\u2029"...)
			}
			encoded = encoded[i+6:]
			i = -1
			continue
		}
		i++ // Skip the escaped character, including a literal backslash.
	}
	return append(result, encoded...), nil
}
