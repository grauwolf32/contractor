package projectsitemap

import "github.com/grauwolf32/contractor/internal/strictjson"

// EditorMutation is an untrusted manual command. Producer and source identity
// belong to the authenticated owner/allocation, never to this wire document.
type EditorMutation struct {
	Operation        string              `json:"operation" strictjson:"required"`
	ExpectedRevision string              `json:"expectedRevision,omitempty"`
	URL              string              `json:"url,omitempty"`
	Method           string              `json:"method,omitempty"`
	MemberIDs        []string            `json:"memberIds,omitempty"`
	Note             string              `json:"note,omitempty"`
	Tags             []string            `json:"tags,omitempty"`
	MarkerID         string              `json:"markerId,omitempty"`
	Source           *EditorSourceUpdate `json:"source,omitempty"`
}

type EditorSourceUpdate struct {
	Expected *Source      `json:"expected,omitempty"`
	Rules    []EditorRule `json:"rules" strictjson:"required"`
}

type EditorRule struct {
	Origin     string                `json:"origin" strictjson:"required"`
	Method     string                `json:"method" strictjson:"required"`
	Path       string                `json:"path" strictjson:"required"`
	Pointer    string                `json:"pointer" strictjson:"required"`
	Parameters []string              `json:"parameters" strictjson:"required"`
	Structural []StructuralParameter `json:"structural" strictjson:"required"`
}

func (m *EditorMutation) UnmarshalJSON(data []byte) error {
	type wire EditorMutation
	var value wire
	if err := strictjson.DecodeNonNull(data, &value); err != nil {
		return ErrInvalid
	}
	*m = EditorMutation(value)
	return nil
}

// WithSource binds a decoded editor command to trusted authority. Expected
// source descriptors remain unchanged so the reducer checks their precondition.
func (m EditorMutation) WithSource(sourceID string) Mutation {
	result := Mutation{
		Operation: m.Operation, ExpectedRevision: m.ExpectedRevision,
		URL: m.URL, Method: m.Method, MemberIDs: m.MemberIDs,
		Note: m.Note, Tags: m.Tags, MarkerID: m.MarkerID,
	}
	if m.Source != nil {
		rules := make([]Rule, len(m.Source.Rules))
		for i, rule := range m.Source.Rules {
			rules[i] = Rule{SourceID: sourceID, Origin: rule.Origin, Method: rule.Method,
				Path: rule.Path, Pointer: rule.Pointer, Parameters: rule.Parameters, Structural: rule.Structural}
		}
		result.Source = &SourceUpdate{SourceID: sourceID, Expected: m.Source.Expected, Rules: rules}
	}
	return result
}
