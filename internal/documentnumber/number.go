// Package documentnumber normalizes the JSON-compatible numbers accepted by
// OpenAPI source parsers without losing the exact safe-integer boundary.
package documentnumber

import (
	"math"
	"regexp"
	"strconv"
	"strings"
)

const maximum = 1<<53 - 1

var numericYAMLScalar = regexp.MustCompile(`^[+-]?(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9][0-9_]*)(?:[eE][+-]?[0-9]+)?)$`)

// LooksNumeric catches plain YAML scalars that the resolver left as strings
// only because the numeric value overflowed its built-in representation.
func LooksNumeric(text string) bool {
	return numericYAMLScalar.MatchString(text)
}

func Integer(text string) (float64, bool) {
	value, err := strconv.ParseInt(strings.ReplaceAll(text, "_", ""), 0, 64)
	if err != nil || value < -maximum || value > maximum {
		return 0, false
	}
	return float64(value), true
}

// Float checks the exact decimal magnitude before accepting the float64.
// Comparing only the rounded value would admit 9007199254740991.1.
func Float(text string) (float64, bool) {
	if !numericYAMLScalar.MatchString(text) {
		return 0, false
	}
	value, err := strconv.ParseFloat(text, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value) > maximum {
		return 0, false
	}
	abs := strings.TrimPrefix(strings.TrimPrefix(text, "+"), "-")
	mantissa, exponentText, _ := strings.Cut(strings.ToLower(abs), "e")
	integer, fraction, _ := strings.Cut(mantissa, ".")
	digits := strings.TrimLeft(integer+fraction, "0")
	if digits == "" {
		return value, true
	}
	// Underflow would silently turn an explicit nonzero example into zero.
	if value == 0 {
		return 0, false
	}
	exponent := int64(0)
	if exponentText != "" {
		exponent, err = strconv.ParseInt(exponentText, 10, 32)
		if err != nil {
			return 0, false
		}
	}
	integerDigits := int64(len(digits)-len(fraction)) + exponent
	if integerDigits > 16 {
		return 0, false
	}
	if integerDigits == 16 {
		whole := digits
		if len(whole) < 16 {
			whole += strings.Repeat("0", 16-len(whole))
		} else {
			whole = whole[:16]
		}
		if whole > "9007199254740991" ||
			(whole == "9007199254740991" && len(digits) > 16 && strings.Trim(digits[16:], "0") != "") {
			return 0, false
		}
	}
	return value, true
}
