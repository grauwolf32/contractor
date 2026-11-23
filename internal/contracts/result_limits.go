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
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return 0, err
	}
	encoded := buffer.Bytes()
	size := len(encoded) - 1 // Encoder adds a newline after the JSON value.
	for i := 0; i+1 < len(encoded); i++ {
		if encoded[i] != '\\' {
			continue
		}
		if i+6 <= len(encoded) && (string(encoded[i:i+6]) == `\u2028` || string(encoded[i:i+6]) == `\u2029`) {
			size -= 3 // Six ASCII escape bytes represent three UTF-8 bytes.
		}
		i++ // Skip the escaped character, including a literal backslash.
	}
	return size, nil
}
