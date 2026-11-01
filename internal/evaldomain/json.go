package evaldomain

import (
	"encoding/json"
	"errors"
	"math"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
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

// strictJSONBounds applies MaxDepth below the top-level value, which
// DecodeBounded counts as depth 1. Every value takes at least one byte of its
// own, so MaxDocumentBytes nodes never bind before the document size does.
var strictJSONBounds = strictjson.Bounds{MaxDepth: MaxDepth + 1, MaxNodes: MaxDocumentBytes}

// StrictJSON retains JSON numbers and rejects ambiguities before schema decode.
// Invalid JSON is eval_invalid even when it is also too deep; within a valid
// document the first violation decides between eval_limit_exceeded (depth)
// and eval_invalid (duplicate member name or non-finite number). A lone
// UTF-16 surrogate escape or the escape \u0000 in any string or member name
// is eval_invalid too (EVF-R236): PostgreSQL cannot decode U+0000 as jsonb,
// so a stored Eval document that held it would break every read that casts
// it.
func StrictJSON(data []byte) (any, error) {
	return strictJSON(data, true)
}

// StrictOutputJSON is the json@1 structural check of a Run output: the rules
// of StrictJSON except the NUL rule. An output is an artifact that no Eval
// document embeds and no Eval read casts, and the native policy digest pins
// the verdicts json@1 gave before EVF-R236, so a NUL escape in an output
// still passes.
func StrictOutputJSON(data []byte) (any, error) {
	return strictJSON(data, false)
}

func strictJSON(data []byte, document bool) (any, error) {
	if len(data) > MaxDocumentBytes {
		return nil, Failure("eval_limit_exceeded")
	}
	if len(data) == 0 || !utf8.Valid(data) || !json.Valid(data) || !strictjson.ValidUnicodeEscapes(data) ||
		document && strictjson.HasNULEscape(data) {
		return nil, Failure("eval_invalid")
	}
	value, err := strictjson.DecodeBounded(data, strictJSONBounds, finiteJSONNumber)
	if errors.Is(err, strictjson.ErrBounds) {
		return nil, Failure("eval_limit_exceeded")
	}
	if err != nil {
		return nil, Failure("eval_invalid")
	}
	return value, nil
}

// finiteJSONNumber keeps the literal of every number float64 represents.
func finiteJSONNumber(literal string) (json.Number, bool) {
	number := json.Number(literal)
	value, err := number.Float64()
	return number, err == nil && !math.IsInf(value, 0) && !math.IsNaN(value)
}
