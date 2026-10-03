package evaldomain

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	MaxDocumentBytes = 1 << 20
	MaxDepth         = 32
	MaxMembers       = 10000
	// A native plan embeds each member and its execution-order entry in one
	// 1 MiB document. External registrations retain the portable 10,000 limit.
	MaxNativeMembers = 1000
	MaxRepetitions   = 100
	MaxPageSize      = 100
)

// StrictJSON retains JSON numbers and rejects ambiguities before schema decode.
func StrictJSON(data []byte) (any, error) {
	if len(data) > MaxDocumentBytes {
		return nil, Failure("eval_limit_exceeded")
	}
	if len(data) == 0 || !utf8.Valid(data) || !json.Valid(data) || !strictjson.ValidUnicodeEscapes(data) {
		return nil, Failure("eval_invalid")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	value, err := readValue(d, 0)
	if err != nil {
		return nil, err
	}
	if _, err = d.Token(); err != io.EOF {
		return nil, Failure("eval_invalid")
	}
	return value, nil
}

func readValue(d *json.Decoder, depth int) (any, error) {
	if depth > MaxDepth {
		return nil, Failure("eval_limit_exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return nil, Failure("eval_invalid")
	}
	switch t {
	case json.Delim('{'):
		m := map[string]any{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return nil, Failure("eval_invalid")
			}
			k, ok := key.(string)
			if !ok {
				return nil, Failure("eval_invalid")
			}
			if _, ok = m[k]; ok {
				return nil, Failure("eval_invalid")
			}
			v, err := readValue(d, depth+1)
			if err != nil {
				return nil, err
			}
			m[k] = v
		}
		if end, err := d.Token(); err != nil || end != json.Delim('}') {
			return nil, Failure("eval_invalid")
		}
		return m, nil
	case json.Delim('['):
		a := []any{}
		for d.More() {
			v, err := readValue(d, depth+1)
			if err != nil {
				return nil, err
			}
			a = append(a, v)
		}
		if end, err := d.Token(); err != nil || end != json.Delim(']') {
			return nil, Failure("eval_invalid")
		}
		return a, nil
	default:
		if n, ok := t.(json.Number); ok {
			value, err := n.Float64()
			if err != nil || math.IsInf(value, 0) || math.IsNaN(value) {
				return nil, Failure("eval_invalid")
			}
		}
		return t, nil
	}
}
