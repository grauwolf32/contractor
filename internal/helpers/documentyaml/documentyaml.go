// Package documentyaml holds the one YAML scalar and mapping-key rule of
// JSON-compatible source documents (SCN-R124). Server scan preparation
// (internal/scanplan) and the Audit inventories (internal/auditdomain) read
// YAML with go-yaml v4 and turn its nodes into JSON values through this
// package; the Runtime OpenAPI authoring Toolset openapi@1 ports the same rule
// (contractor_runtime.toolsets.openapi.yaml_scalars), and
// api/testdata/openapi-yaml-scalar-cases.json holds both to the same verdicts.
//
// Anchors, aliases, bounds and duplicate member names stay with each reader;
// a merge key is rejected here, because it is a key.
package documentyaml

import (
	"strings"

	"github.com/grauwolf32/contractor/internal/helpers/documentnumber"
	"go.yaml.in/yaml/v4"
)

// Scalar returns the JSON value of a scalar node.
//
// An untagged plain scalar keeps go-yaml's resolution: null, true and false
// in their core spellings, numbers within the exact integer range of IEEE 754
// doubles (LIM-SCN-67), and a plain date as its source text. A plain string
// spelled like a number that did not resolve to one, because it overflows, is
// rejected rather than read as a string. Quoted and block scalars are
// strings. An explicit tag must be a core scalar tag; !!null, !!bool, !!int
// and !!float accept only text that means a value of that type.
func Scalar(node *yaml.Node) (any, bool) {
	if node.Kind != yaml.ScalarNode || node.Tag == "!" {
		return nil, false
	}
	tagged := node.Style&yaml.TaggedStyle != 0
	switch node.Tag {
	case "!!str":
		// The resolver leaves an overflowing number a string. Quoted and
		// explicitly tagged strings are ordinary data.
		if node.Style == 0 && documentnumber.LooksNumeric(node.Value) {
			return nil, false
		}
		return node.Value, true
	case "!!timestamp":
		// JSON carries dates as strings. Only the resolver may give this tag;
		// an explicit !!timestamp is not a core tag.
		if tagged {
			return nil, false
		}
		return node.Value, true
	case "!!null":
		switch node.Value {
		case "", "~", "null", "Null", "NULL":
			return nil, true
		}
	case "!!bool":
		switch node.Value {
		case "true", "True", "TRUE":
			return true, true
		case "false", "False", "FALSE":
			return false, true
		}
	case "!!int":
		if value, ok := documentnumber.Integer(node.Value); ok {
			return value, true
		}
	case "!!float":
		if value, ok := documentnumber.Float(strings.ReplaceAll(node.Value, "_", "")); ok {
			return value, true
		}
	}
	return nil, false
}

// Key returns the member name of a mapping key node. A plain key is the
// string of its source text whatever it resolves to, so 200:, 1e3:, on: and
// 2023-01-01: name the members "200", "1e3", "on" and "2023-01-01". A quoted
// or block key is its string, an explicitly tagged key must be !!str, and a
// merge key << is rejected.
func Key(node *yaml.Node) (string, bool) {
	if node.Kind != yaml.ScalarNode || node.Tag == "!" {
		return "", false
	}
	if node.Style&yaml.TaggedStyle != 0 && node.Tag != "!!str" || node.Tag == "!!merge" {
		return "", false
	}
	return node.Value, true
}

// Collection reports whether a mapping or sequence node carries its own core
// tag, either resolved or explicit.
func Collection(node *yaml.Node) bool {
	switch node.Kind {
	case yaml.MappingNode:
		return node.Tag == "!!map"
	case yaml.SequenceNode:
		return node.Tag == "!!seq"
	}
	return false
}
