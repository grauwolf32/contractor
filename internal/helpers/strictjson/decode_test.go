package strictjson

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

type decodeInner struct {
	Name  string `json:"name"`
	Count int    `json:"count,omitempty"`
}

type decodeOuter struct {
	Kind     string                 `json:"kind"`
	Inner    decodeInner            `json:"inner"`
	Optional *decodeInner           `json:"optional,omitempty"`
	Items    []decodeInner          `json:"items,omitempty"`
	ByName   map[string]decodeInner `json:"byName,omitempty"`
	Anything any                    `json:"anything,omitempty"`
	Raw      json.RawMessage        `json:"raw,omitempty"`
	Self     selfDecoding           `json:"self,omitempty"`
	Nullable *string                `json:"nullable"`
}

// selfDecoding owns its members, as the contracts types with UnmarshalJSON do.
type selfDecoding struct{ Value string }

func (s *selfDecoding) UnmarshalJSON(data []byte) error {
	var wire struct {
		Value string `json:"value"`
	}
	if err := Decode(data, &wire); err != nil {
		return err
	}
	s.Value = wire.Value
	return nil
}

const validOuter = `{"kind":"k","inner":{"name":"a"},"nullable":null}`

func decodeOuterJSON(data string) error {
	var value decodeOuter
	return Decode([]byte(data), &value)
}

func TestDecodeAcceptsExactMembers(t *testing.T) {
	var value decodeOuter
	input := `{"kind":"k","inner":{"name":"a","count":2},"optional":{"name":"b"},` +
		`"items":[{"name":"c"}],"byName":{"x":{"name":"d"},"X":{"name":"e"}},` +
		`"anything":{"a":1,"A":2},"raw":{"b":1},"self":{"value":"v"},"nullable":"n"}`
	if err := Decode([]byte(input), &value); err != nil {
		t.Fatalf("Decode = %v", err)
	}
	if value.Inner.Count != 2 || value.Optional.Name != "b" || value.ByName["X"].Name != "e" ||
		value.Self.Value != "v" || *value.Nullable != "n" {
		t.Fatalf("decoded = %+v", value)
	}
	if err := decodeOuterJSON(validOuter); err != nil {
		t.Fatalf("Decode(%s) = %v", validOuter, err)
	}
}

func TestDecodeRejectsDuplicateKeysEverywhere(t *testing.T) {
	for _, input := range []string{
		`{"kind":"k","kind":"k","inner":{"name":"a"},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a","name":"b"},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"items":[{"name":"a","count":1,"count":2}],"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"byName":{"x":{"name":"a"},"x":{"name":"b"}},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"anything":[{"a":1,"a":2}],"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"raw":{"a":{"b":1,"b":1}},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"self":{"value":"v","value":"w"},"nullable":null}`,
		// Escaped and plain spellings of one name are one name.
		`{"kind":"k","kind":"k","inner":{"name":"a"},"nullable":null}`,
		// An unknown member repeated is still a duplicate first.
		`{"kind":"k","inner":{"name":"a"},"Other":1,"Other":2,"nullable":null}`,
	} {
		if err := decodeOuterJSON(input); !errors.Is(err, ErrDuplicateKey) {
			t.Errorf("Decode(%s) = %v, want duplicate key", input, err)
		}
	}
}

func TestDecodeRejectsCaseAliasesAndUnknownMembers(t *testing.T) {
	for _, input := range []string{
		`{"Kind":"k","inner":{"name":"a"},"nullable":null}`,
		`{"kind":"k","INNER":{"name":"a"},"nullable":null}`,
		`{"kind":"k","inner":{"Name":"a"},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"items":[{"name":"a","COUNT":1}],"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"byName":{"x":{"nAme":"a"}},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"optional":{"name":"a","extra":1},"nullable":null}`,
		`{"kind":"k","inner":{"name":"a"},"self":{"Value":"v"},"nullable":null}`,
		// The Kelvin sign folds to k in encoding/json.
		"{\"Kind\":\"k\",\"inner\":{\"name\":\"a\"},\"nullable\":null}",
	} {
		if err := decodeOuterJSON(input); !errors.Is(err, ErrUnknownField) {
			t.Errorf("Decode(%s) = %v, want unknown member", input, err)
		}
	}
	err := decodeOuterJSON(`{"kind":"k","inner":{"name":"a"},"Nullable":null}`)
	if err == nil || !strings.Contains(err.Error(), `"Nullable" differs in case from "nullable"`) {
		t.Fatalf("case alias error = %v", err)
	}
}

func TestDecodeReportsDuplicateKeysBeforeMemberErrors(t *testing.T) {
	input := `{"Kind":"k","inner":{"name":"a"},"items":[{"name":"a","name":"a"}],"nullable":null}`
	if err := decodeOuterJSON(input); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("Decode = %v, want duplicate key", err)
	}
	if err := decodeOuterJSON(`{"Kind":"k","inner":{"name":"a"},"nullable":null} {}`); !errors.Is(err, ErrTrailingData) {
		t.Fatalf("Decode = %v, want trailing data", err)
	}
}

func TestDecodeKeepsEncodingJSONErrors(t *testing.T) {
	var value decodeOuter
	err := Decode([]byte(`{"kind":"k","inner":{"name":"a"},"nullable":null,}`), &value)
	var syntax *json.SyntaxError
	if !errors.As(err, &syntax) {
		t.Fatalf("syntax error = %T %v", err, err)
	}
	err = Decode([]byte(`{"kind":1,"inner":{"name":"a"},"nullable":null}`), &value)
	var typeErr *json.UnmarshalTypeError
	if !errors.As(err, &typeErr) {
		t.Fatalf("type error = %T %v", err, err)
	}
	if err := Decode([]byte(``), &value); err == nil {
		t.Fatal("empty input accepted")
	}
	if err := Decode([]byte(`{}`), value); err == nil {
		t.Fatal("non-pointer target accepted")
	}
	deep := strings.Repeat("[", maxDepth+1) + strings.Repeat("]", maxDepth+1)
	var anything any
	if err := Decode([]byte(deep), &anything); err == nil {
		t.Fatal("input beyond the nesting limit accepted")
	}
}

type promotedBase struct {
	ID     string `json:"id"`
	Hidden string `json:"hidden"`
}

type promotedOther struct {
	Shared string `json:"shared"`
}

type promotedThird struct {
	Shared string `json:"shared"`
}

type promoted struct {
	promotedBase
	*promotedOther
	promotedThird
	Hidden   string `json:"hidden"`
	Untagged string
	Skipped  string `json:"-"`
	Dash     string `json:"-,"`
	private  string
}

func TestDecodeFollowsEncodingJSONFieldRules(t *testing.T) {
	var value promoted
	if err := Decode([]byte(`{"id":"i","hidden":"h","Untagged":"u","-":"d"}`), &value); err != nil {
		t.Fatalf("Decode = %v", err)
	}
	if value.ID != "i" || value.Hidden != "h" || value.promotedBase.Hidden != "" || value.Untagged != "u" ||
		value.Dash != "d" || value.private != "" {
		t.Fatalf("decoded = %+v", value)
	}
	for input, want := range map[string]error{
		`{"id":"i","shared":"s"}`:              ErrUnknownField, // two shared fields at one depth annihilate
		`{"id":"i","Skipped":"s"}`:             ErrUnknownField,
		`{"id":"i","untagged":"u"}`:            ErrUnknownField,
		`{"id":"i","private":"p"}`:             ErrUnknownField,
		`{"id":"i","promotedBase":{}}`:         ErrUnknownField,
		`{"id":"i","hidden":"a","hidden":"b"}`: ErrDuplicateKey,
	} {
		var target promoted
		if err := Decode([]byte(input), &target); !errors.Is(err, want) {
			t.Errorf("Decode(%s) = %v, want %v", input, err, want)
		}
	}
}

type recursive struct {
	Name     string       `json:"name"`
	Children []*recursive `json:"children,omitempty"`
	Next     *recursive   `json:"next,omitempty"`
}

func TestDecodeChecksRecursiveTypes(t *testing.T) {
	var value recursive
	if err := Decode([]byte(`{"name":"a","children":[{"name":"b","next":{"name":"c"}}]}`), &value); err != nil {
		t.Fatalf("Decode = %v", err)
	}
	if err := Decode([]byte(`{"name":"a","children":[{"name":"b","next":{"Name":"c"}}]}`), &value); !errors.Is(err, ErrUnknownField) {
		t.Fatalf("Decode = %v", err)
	}
}

type concurrentTarget struct {
	Left  *concurrentTarget `json:"left"`
	Right []concurrentLeaf  `json:"right"`
}

type concurrentLeaf struct {
	Value int `json:"value"`
}

func TestDecodeBuildsTypeInformationOnceUnderConcurrency(t *testing.T) {
	var group sync.WaitGroup
	for range 16 {
		group.Add(1)
		go func() {
			defer group.Done()
			var value concurrentTarget
			if err := Decode([]byte(`{"left":{"left":null},"right":[{"value":1}]}`), &value); err != nil {
				t.Errorf("Decode = %v", err)
			}
			if err := Decode([]byte(`{"left":null,"right":[{"Value":1}]}`), &value); !errors.Is(err, ErrUnknownField) {
				t.Errorf("Decode = %v", err)
			}
		}()
	}
	group.Wait()
}

func TestDecodeChecksLargeObjects(t *testing.T) {
	var builder strings.Builder
	builder.WriteString("{")
	for index := range 100 {
		if index > 0 {
			builder.WriteString(",")
		}
		builder.WriteString(`"k`)
		builder.WriteString(strings.Repeat("x", index))
		builder.WriteString(`":1`)
	}
	unique := builder.String() + "}"
	duplicate := builder.String() + `,"kxxxxx":2}`
	var value map[string]int
	if err := Decode([]byte(unique), &value); err != nil || len(value) != 100 {
		t.Fatalf("Decode = %v, %d", err, len(value))
	}
	if err := Decode([]byte(duplicate), &value); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("Decode = %v", err)
	}
}

// FuzzScannerMatchesEncodingJSONSyntax holds the scanner to encoding/json's
// grammar: it must accept every value encoding/json accepts and reject the
// rest.
func FuzzScannerMatchesEncodingJSONSyntax(f *testing.F) {
	for _, seed := range []string{
		`{}`, `[]`, `""`, `0`, `-0.5e+10`, `true`, `null`, `{"a":[1,{"b":"é\n"}]}`,
		`{"a":1,"a":2}`, `{"a":1} x`, `[1,]`, `{"a"}`, `"\ud800"`, "\"\xff\"", `01`, `1.`, `"\x"`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		err := check(data, anyInfo)
		switch {
		case err == nil && !json.Valid(data):
			t.Fatalf("scanner accepted invalid JSON %q", data)
		case errors.Is(err, errSyntax) && json.Valid(data):
			t.Fatalf("scanner rejected valid JSON %q", data)
		case errors.Is(err, ErrTrailingData) && json.Valid(data):
			t.Fatalf("scanner found trailing data in one value %q", data)
		}
	})
}
