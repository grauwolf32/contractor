package evaldomain

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"strconv"
	"unicode/utf8"
)

const (
	MaxDocumentBytes = 1 << 20
	MaxDepth         = 32
	MaxMembers       = 10000
	MaxCases         = 1000
	MaxRepetitions   = 100
	MaxPageSize      = 100
)

// StrictJSON retains JSON numbers and rejects ambiguities before schema decode.
func StrictJSON(data []byte) (any, error) {
	if len(data) > MaxDocumentBytes {
		return nil, Failure("eval_limit_exceeded")
	}
	if len(data) == 0 || !utf8.Valid(data) || !json.Valid(data) || !validEscapes(data) {
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

// encoding/json accepts lone UTF-16 surrogate escapes by replacement. Exact
// byte identities must not silently merge such malformed strings or keys.
func validEscapes(data []byte) bool {
	for i := 0; i < len(data); i++ {
		if data[i] != '\\' {
			continue
		}
		i++
		if i >= len(data) {
			return false
		}
		if data[i] != 'u' {
			continue
		}
		if i+4 >= len(data) {
			return false
		}
		n, err := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
		if err != nil {
			return false
		}
		i += 4
		if n >= 0xDC00 && n <= 0xDFFF {
			return false
		}
		if n < 0xD800 || n > 0xDBFF {
			continue
		}
		if i+6 >= len(data) || data[i+1] != '\\' || data[i+2] != 'u' {
			return false
		}
		low, err := strconv.ParseUint(string(data[i+3:i+7]), 16, 16)
		if err != nil || low < 0xDC00 || low > 0xDFFF {
			return false
		}
		i += 6
	}
	return true
}
