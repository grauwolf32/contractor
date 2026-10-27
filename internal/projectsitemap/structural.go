package projectsitemap

import (
	"bytes"
	"net/url"
	"slices"
	"strings"
)

func canonicalRules(rules []Rule) []Rule {
	copy := append([]Rule{}, rules...)
	for i := range copy {
		r := &copy[i]
		r.Parameters = append([]string{}, r.Parameters...)
		slices.Sort(r.Parameters)
		r.Structural = append([]StructuralParameter{}, r.Structural...)
		for j := range r.Structural {
			r.Structural[j].Allowed = append([]string{}, r.Structural[j].Allowed...)
			slices.Sort(r.Structural[j].Allowed)
		}
		slices.SortFunc(r.Structural, func(a, b StructuralParameter) int { return strings.Compare(a.Name, b.Name) })
	}
	slices.SortFunc(copy, func(a, b Rule) int { return bytes.Compare(jsonBytes(a), jsonBytes(b)) })
	return copy
}

func groupingPolicyDigest(rules []Rule) string {
	if len(rules) == 0 {
		return key(Projection)
	}
	return key(Projection, string(jsonBytes(canonicalRules(rules))))
}

func sameStructuralNames(rules []Rule, origin, method, route string, current []StructuralValue) bool {
	names := map[string]bool{}
	for _, r := range rules {
		if r.Origin == origin && r.Method == method && r.Path == route {
			for _, s := range r.Structural {
				names[s.Name] = true
			}
		}
	}
	if len(names) != len(current) {
		return false
	}
	for _, s := range current {
		if !names[s.Name] {
			return false
		}
	}
	return true
}

func validStructural(values []StructuralValue) bool {
	if len(values) > MaxTags {
		return false
	}
	names := map[string]bool{}
	for _, value := range values {
		if names[value.Name] || !validLabel(value.Name) || len(value.Values) > MaxTags {
			return false
		}
		names[value.Name] = true
		for _, item := range value.Values {
			if len(item) > MaxTagBytes || !validText(item) {
				return false
			}
		}
	}
	return true
}

// projectStructural retains explicitly configured finite values from the query.
// Conflicting policies and unknown values omit the whole concrete identity
// instead of merging under a made-up placeholder.
func projectStructural(rules []Rule, origin, method, route, raw string) ([]StructuralValue, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, ErrInvalid
	}
	query, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return nil, ErrInvalid
	}
	selectors, conflict := structuralSelectors(rules, origin, method, route)
	if conflict != nil {
		return nil, errStructuralSelection
	}
	names := make([]string, 0, len(selectors))
	for name := range selectors {
		names = append(names, name)
	}
	slices.Sort(names)
	result := []StructuralValue{}
	for _, name := range names {
		values := query[name]
		if len(values) > MaxTags {
			return nil, errStructuralSelection
		}
		for _, value := range values {
			if !slices.Contains(selectors[name], value) {
				return nil, errStructuralSelection
			}
		}
		result = append(result, StructuralValue{Name: name, Values: append([]string{}, values...)})
	}
	return result, nil
}

// structuralSelectors returns the effective finite selector policy or one pair
// that proves a conflict. A conflict omits the operation even when the query
// contains neither selector; receipts must preserve that distinction.
func structuralSelectors(rules []Rule, origin, method, route string) (map[string][]string, []StructuralParameter) {
	selectors := map[string][]string{}
	for _, rule := range rules {
		if rule.Origin != origin || rule.Method != method || rule.Path != route {
			continue
		}
		for _, selector := range rule.Structural {
			allowed := slices.Clone(selector.Allowed)
			slices.Sort(allowed)
			if existing, found := selectors[selector.Name]; found && !slices.Equal(existing, allowed) {
				return nil, []StructuralParameter{
					{Name: selector.Name, Allowed: existing},
					{Name: selector.Name, Allowed: allowed},
				}
			}
			selectors[selector.Name] = allowed
		}
	}
	return selectors, nil
}
