package strictjson

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

type requiredInner struct {
	Name  string `json:"name" strictjson:"required"`
	Count int    `json:"count,omitempty"`
}

type requiredOuter struct {
	Flag     bool                     `json:"flag" strictjson:"required"`
	Inner    requiredInner            `json:"inner" strictjson:"required"`
	Maybe    *requiredInner           `json:"maybe" strictjson:"required,nullable"`
	Optional *requiredInner           `json:"optional,omitempty"`
	ByName   map[string]requiredInner `json:"byName,omitempty"`
	Items    []requiredInner          `json:"items,omitempty"`
	Pointers []*requiredInner         `json:"pointers,omitempty"`
	Self     selfDecoding             `json:"self,omitempty"`
}

type requiredEmbedding struct {
	requiredInner
	Extra string `json:"extra"`
}

func TestDecodeAcceptsEveryRequiredMember(t *testing.T) {
	for _, input := range []string{
		`{"flag":false,"inner":{"name":""},"maybe":null}`,
		`{"maybe":{"name":"m"},"inner":{"name":"a","count":1},"flag":true,"optional":null,` +
			`"byName":{"x":{"name":"b"}},"items":[{"name":"c"}],"pointers":[null,{"name":"d"}],"self":null}`,
		`{"flag":true,"inner":{"name":"a","count":null},"maybe":{"name":"m"},"self":{"value":"v"}}`,
	} {
		var value requiredOuter
		if err := Decode([]byte(input), &value); err != nil {
			t.Errorf("Decode(%s) = %v", input, err)
		}
	}
	var pointer *requiredOuter
	if err := Decode([]byte(`null`), &pointer); err != nil || pointer != nil {
		t.Fatalf("Decode(null) into a pointer = %v, %v", err, pointer)
	}
}

func TestDecodeRejectsOmittedOrNullRequiredMembers(t *testing.T) {
	for input, want := range map[string]string{
		`{"inner":{"name":"a"},"maybe":null}`:                           `"flag"`,
		`{"flag":true,"maybe":null}`:                                    `"inner"`,
		`{"flag":true,"inner":{"name":"a"}}`:                            `"maybe"`,
		`{}`:                                                            `"flag"`,
		`{"flag":true,"inner":{},"maybe":null}`:                         `"name"`,
		`{"flag":true,"inner":{"name":"a"},"maybe":{"count":1}}`:        `"name"`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"optional":{}}`: `"name"`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"byName":{"x":{"count":1}}}`: `"name"`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"items":[{"name":"b"},{}]}`:  `"name"`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"pointers":[{"count":2}]}`:   `"name"`,
		`{"flag":null,"inner":{"name":"a"},"maybe":null}`:                            `"flag" is null`,
		`{"flag":true,"inner":null,"maybe":null}`:                                    `"inner" is null`,
		`{"flag":true,"inner":{"name":null},"maybe":null}`:                           `"name" is null`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"optional":{"name":null}}`:   `"name" is null`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"byName":{"x":null}}`:        `null instead of an object`,
		`{"flag":true,"inner":{"name":"a"},"maybe":null,"items":[null]}`:             `null instead of an object`,
		`null`: `null instead of an object`,
	} {
		var value requiredOuter
		if err := Decode([]byte(input), &value); !errors.Is(err, ErrMissingField) || !strings.Contains(err.Error(), want) {
			t.Errorf("Decode(%s) = %v, want missing %s", input, err, want)
		}
	}
	var embedding requiredEmbedding
	if err := Decode([]byte(`{"extra":"e"}`), &embedding); !errors.Is(err, ErrMissingField) {
		t.Fatalf("promoted required member: Decode = %v", err)
	}
}

func TestDecodeReportsDuplicateAndUnknownMembersBeforeMissingOnes(t *testing.T) {
	var value requiredOuter
	if err := Decode([]byte(`{"flag":true,"flag":true}`), &value); !errors.Is(err, ErrDuplicateKey) {
		t.Fatalf("Decode = %v, want duplicate key", err)
	}
	if err := Decode([]byte(`{"Flag":true,"inner":{"name":"a"},"maybe":null}`), &value); !errors.Is(err, ErrUnknownField) {
		t.Fatalf("Decode = %v, want unknown member", err)
	}
}

func TestDecodeRequiresMembersBeyondTheFirstSixtyFour(t *testing.T) {
	fields := make([]reflect.StructField, 70)
	for index := range fields {
		fields[index] = reflect.StructField{
			Name: fmt.Sprintf("F%02d", index), Type: reflect.TypeFor[int](),
			Tag: reflect.StructTag(fmt.Sprintf(`json:"f%02d"`, index)),
		}
	}
	fields[69].Tag = `json:"f69" strictjson:"required"`
	target := reflect.New(reflect.StructOf(fields)).Interface()
	if err := Decode([]byte(`{"f68":1,"f69":2}`), target); err != nil {
		t.Fatalf("Decode = %v", err)
	}
	if err := Decode([]byte(`{"f68":1,"f00":2}`), target); !errors.Is(err, ErrMissingField) {
		t.Fatalf("Decode = %v, want missing f69", err)
	}
}

type invalidStrictTag struct {
	Value int `json:"value" strictjson:"required,unknown"`
}

func TestDecodeRejectsAnInvalidStrictTag(t *testing.T) {
	var value invalidStrictTag
	err := Decode([]byte(`{"value":1}`), &value)
	if err == nil || errors.Is(err, ErrMissingField) || !strings.Contains(err.Error(), `invalid strictjson tag "required,unknown"`) {
		t.Fatalf("Decode = %v", err)
	}
}
