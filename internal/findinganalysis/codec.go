package findinganalysis

import (
	"bytes"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/documentnumber"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

var errInvalid error = &auditdomain.ValidationError{Code: auditdomain.CodeInvalid, Field: "finding_analysis"}
var identifier = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$`)
var ruleIdentifier = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$`)

// decode requires the canonical bytes authors publish, including strict member
// presence and non-null optional members; it never repairs malformed claims.
func decode(data []byte) (Document, error) {
	var value Document
	if len(data) > maximumBytes || !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) || strictjson.HasNULEscape(data) {
		return value, errInvalid
	}
	if _, err := strictjson.DecodeBounded(data, strictjson.Bounds{MaxDepth: 32, MaxNodes: 100000, MaxStringBytes: maximumTextBytes}, documentnumber.Float); err != nil {
		return value, errInvalid
	}
	if strictjson.DecodeNonNull(data, &value) != nil {
		return value, errInvalid
	}
	canonical, err := strictjson.Canonical(value)
	if err != nil || !bytes.Equal(canonical, data) || validate(value) != nil {
		return Document{}, errInvalid
	}
	return value, nil
}

func text(value string, required bool) bool {
	return utf8.ValidString(value) && len(value) <= maximumTextBytes && !strings.ContainsRune(value, 0) && (!required || strings.TrimSpace(value) != "")
}

func validPath(value string) bool {
	if !text(value, true) || len(value) > 512 || strings.ContainsAny(value, "\\:") || strings.IndexFunc(value, unicode.IsControl) >= 0 {
		return false
	}
	for _, part := range strings.Split(value, "/") {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

func validate(value Document) error {
	if value.Schema != Schema || !identifier.MatchString(value.ClientKey) || value.Sources == nil || len(value.Sources) > 32 || value.Locations == nil || len(value.Locations) > 256 || value.Flows == nil || len(value.Flows) > 32 || value.HTTPExchanges == nil || len(value.HTTPExchanges) > 32 {
		return errInvalid
	}
	if rule := value.Rule; rule != nil {
		if !ruleIdentifier.MatchString(rule.ID) || !text(rule.Version, true) || !text(rule.Name, true) || !text(rule.Description, true) {
			return errInvalid
		}
		if rule.HelpURI != "" {
			u, err := url.Parse(rule.HelpURI)
			if err != nil || !text(rule.HelpURI, true) || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.Opaque != "" || strings.IndexFunc(rule.HelpURI, unicode.IsSpace) >= 0 {
				return errInvalid
			}
		}
	}
	sources := map[string]bool{}
	for _, s := range value.Sources {
		if !identifier.MatchString(s.ID) || sources[s.ID] || !slices.Contains([]string{"archive", "file"}, s.Kind) || s.ArtifactRef.ValidateExact() != nil {
			return errInvalid
		}
		sources[s.ID] = true
	}
	locations := map[string]bool{}
	for _, l := range value.Locations {
		if !identifier.MatchString(l.ID) || locations[l.ID] || !sources[l.SourceID] || !validPath(l.Path) || !contentdigest.Valid(l.FileDigest) {
			return errInvalid
		}
		locations[l.ID] = true
		if r := l.Region; r != nil {
			end := r.StartLine
			if r.EndLine != nil {
				end = *r.EndLine
			}
			if r.StartLine < 1 || r.StartLine > 1<<53-1 || end < r.StartLine || end > 1<<53-1 || r.EndColumn != nil && r.StartColumn == nil || r.StartColumn != nil && (*r.StartColumn < 1 || *r.StartColumn > 1<<53-1) || r.EndColumn != nil && (*r.EndColumn < 1 || *r.EndColumn > 1<<53-1 || end == r.StartLine && *r.EndColumn < *r.StartColumn) {
				return errInvalid
			}
		}
		if s := l.Symbol; s != nil {
			if !text(s.Name, true) || !text(s.QualifiedName, false) || !text(s.Language, false) || !text(s.Kind, false) || !text(s.Signature, false) {
				return errInvalid
			}
		}
	}
	if value.PrimaryLocationID != "" && !locations[value.PrimaryLocationID] {
		return errInvalid
	}
	flows := map[string]bool{}
	for _, f := range value.Flows {
		if !identifier.MatchString(f.ID) || flows[f.ID] || !slices.Contains([]string{"call-path", "data-flow", "execution-trace"}, f.Kind) || !slices.Contains([]string{"complete", "partial"}, f.Completeness) || !text(f.Description, true) || len(f.Steps) < 1 || len(f.Steps) > 128 || f.Edges == nil || f.Gaps == nil || f.EvidenceRefs == nil || len(f.Edges)+len(f.Gaps) != len(f.Steps)-1 {
			return errInvalid
		}
		flows[f.ID] = true
		steps := map[string]int{}
		for i, s := range f.Steps {
			if !identifier.MatchString(s.ID) || !text(s.Message, true) || !text(s.Value, false) || s.EvidenceRefs == nil || s.LocationID != "" && !locations[s.LocationID] || len(s.Roles) == 0 || len(s.Roles) > 8 || s.ValueState != "" && !slices.Contains([]string{"unknown", "tainted", "derived", "validated", "clean"}, s.ValueState) {
				return errInvalid
			}
			if _, exists := steps[s.ID]; exists {
				return errInvalid
			}
			steps[s.ID] = i
			roles := map[string]bool{}
			for _, r := range s.Roles {
				if roles[r] || !slices.Contains([]string{"source", "propagation", "call", "return", "validation", "sanitizer", "sink", "observation"}, r) {
					return errInvalid
				}
				roles[r] = true
			}
		}
		pairs := map[int]bool{}
		pair := func(from, to string, gap bool) bool {
			a, ok := steps[from]
			b, ok2 := steps[to]
			if !ok || !ok2 || b != a+1 || pairs[a] || !gap && (f.Steps[a].LocationID == "" || f.Steps[b].LocationID == "") {
				return false
			}
			pairs[a] = true
			return true
		}
		for _, e := range f.Edges {
			if !pair(e.From, e.To, false) || !slices.Contains([]string{"call", "return", "argument", "assignment", "transform", "control", "observation"}, e.Relation) || !slices.Contains([]string{"producer-asserted", "tool-reported"}, e.Basis) || len(e.EvidenceRefs) == 0 {
				return errInvalid
			}
			if b := e.ArgumentBinding; b != nil {
				if e.Relation != "argument" || !text(b.CallerValue, true) || !text(b.CalleeParameter, true) {
					return errInvalid
				}
			}
			if f.Kind == "data-flow" && e.Relation == "argument" && e.ArgumentBinding == nil {
				return errInvalid
			}
		}
		for _, g := range f.Gaps {
			if !pair(g.AfterStepID, g.BeforeStepID, true) || !text(g.Reason, true) {
				return errInvalid
			}
		}
		if f.Completeness == "complete" {
			if len(f.Gaps) != 0 {
				return errInvalid
			}
			if f.Kind == "data-flow" && (f.Steps[0].LocationID == "" || f.Steps[len(f.Steps)-1].LocationID == "" || !slices.Contains(f.Steps[0].Roles, "source") || !slices.Contains(f.Steps[len(f.Steps)-1].Roles, "sink")) {
				return errInvalid
			}
		}
	}
	exchanges := map[string]bool{}
	for _, h := range value.HTTPExchanges {
		if !identifier.MatchString(h.ID) || exchanges[h.ID] || h.EvidenceRef.ValidateExact() != nil || h.ImplementationLocationID != "" && !locations[h.ImplementationLocationID] {
			return errInvalid
		}
		exchanges[h.ID] = true
		if o := h.Operation; o != nil {
			if o.DocumentRef.ValidateExact() != nil || !strings.HasPrefix(o.Pointer, "/") || !text(o.Pointer, true) || !text(o.Method, true) || !strings.HasPrefix(o.PathTemplate, "/") || !text(o.PathTemplate, true) {
				return errInvalid
			}
		}
	}
	if value.PrimaryHTTPExchangeID != "" && !exchanges[value.PrimaryHTTPExchangeID] {
		return errInvalid
	}
	return nil
}
