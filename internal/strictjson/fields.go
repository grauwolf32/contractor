package strictjson

import (
	"cmp"
	"encoding"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"sync"
	"unicode"
)

// kind says how the scanner checks a JSON value decoded into a Go type.
type kind uint8

const (
	// kindAny checks duplicate keys only: interfaces, scalars and types that
	// decode themselves through UnmarshalJSON or UnmarshalText.
	kindAny kind = iota
	kindPointer
	kindStruct
	kindMap
	kindSequence
)

// typeInfo is the scanner's view of one Go type. Infos form a graph that may
// contain cycles through pointers, maps and slices.
type typeInfo struct {
	kind kind
	// elem describes the pointer target, map value or sequence element.
	elem   *typeInfo
	fields *structInfo
}

type structInfo struct {
	byName  map[string]int
	members []member
}

type member struct {
	name string
	info *typeInfo
}

var (
	anyInfo   = &typeInfo{kind: kindAny}
	infoCache sync.Map // reflect.Type -> *typeInfo
	buildMu   sync.Mutex

	jsonUnmarshalerType = reflect.TypeFor[json.Unmarshaler]()
	textUnmarshalerType = reflect.TypeFor[encoding.TextUnmarshaler]()
)

// infoFor returns the cached checks for t; nil means kindAny.
func infoFor(t reflect.Type) *typeInfo {
	if t == nil {
		return anyInfo
	}
	if cached, ok := infoCache.Load(t); ok {
		return cached.(*typeInfo)
	}
	buildMu.Lock()
	defer buildMu.Unlock()
	building := map[reflect.Type]*typeInfo{}
	info := buildInfo(t, building)
	// Publish the whole graph only once it is complete, so a concurrent
	// reader never observes a partly built type.
	for typ, built := range building {
		infoCache.Store(typ, built)
	}
	return info
}

func buildInfo(t reflect.Type, building map[reflect.Type]*typeInfo) *typeInfo {
	if cached, ok := infoCache.Load(t); ok {
		return cached.(*typeInfo)
	}
	if info, ok := building[t]; ok {
		return info
	}
	info := &typeInfo{kind: kindAny}
	building[t] = info
	if decodesItself(t) {
		return info
	}
	switch t.Kind() {
	case reflect.Pointer:
		info.kind, info.elem = kindPointer, buildInfo(t.Elem(), building)
	case reflect.Map:
		info.kind, info.elem = kindMap, buildInfo(t.Elem(), building)
	case reflect.Slice, reflect.Array:
		info.kind, info.elem = kindSequence, buildInfo(t.Elem(), building)
	case reflect.Struct:
		info.kind, info.fields = kindStruct, buildStruct(t, building)
	}
	return info
}

// decodesItself mirrors encoding/json, which hands a value to UnmarshalJSON
// or UnmarshalText when the type or a pointer to it implements one.
func decodesItself(t reflect.Type) bool {
	if t.Implements(jsonUnmarshalerType) || t.Implements(textUnmarshalerType) {
		return true
	}
	if t.Kind() == reflect.Pointer {
		return false
	}
	pointer := reflect.PointerTo(t)
	return pointer.Implements(jsonUnmarshalerType) || pointer.Implements(textUnmarshalerType)
}

func buildStruct(t reflect.Type, building map[reflect.Type]*typeInfo) *structInfo {
	fields := structFields(t)
	result := &structInfo{byName: make(map[string]int, len(fields)), members: make([]member, len(fields))}
	for index, field := range fields {
		result.members[index] = member{name: field.name, info: buildInfo(field.typ, building)}
		result.byName[field.name] = index
	}
	return result
}

// structField is one member encoding/json decodes into a struct.
type structField struct {
	name   string
	typ    reflect.Type
	tagged bool
	index  []int
}

// structFields returns the members encoding/json decodes into struct type t.
// It follows encoding/json's typeFields: exported and embedded fields, json
// tag names, "-", and the Go visibility rules for promoted fields, where a
// shallower field hides a deeper one and a tagged field wins a tie.
func structFields(t reflect.Type) []structField {
	type embedded struct {
		typ   reflect.Type
		index []int
	}
	var current []embedded
	next := []embedded{{typ: t}}
	var count, nextCount map[reflect.Type]int
	visited := map[reflect.Type]bool{}
	var fields []structField
	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[reflect.Type]int{}
		for _, outer := range current {
			if visited[outer.typ] {
				continue
			}
			visited[outer.typ] = true
			for i := range outer.typ.NumField() {
				sf := outer.typ.Field(i)
				if sf.Anonymous {
					embeddedType := sf.Type
					if embeddedType.Kind() == reflect.Pointer {
						embeddedType = embeddedType.Elem()
					}
					if !sf.IsExported() && embeddedType.Kind() != reflect.Struct {
						continue
					}
				} else if !sf.IsExported() {
					continue
				}
				tag := sf.Tag.Get("json")
				if tag == "-" {
					continue
				}
				name, _, _ := strings.Cut(tag, ",")
				if !validTagName(name) {
					name = ""
				}
				index := append(slices.Clone(outer.index), i)
				followed := sf.Type
				if followed.Name() == "" && followed.Kind() == reflect.Pointer {
					followed = followed.Elem()
				}
				if name != "" || !sf.Anonymous || followed.Kind() != reflect.Struct {
					field := structField{name: name, typ: sf.Type, tagged: name != "", index: index}
					if field.name == "" {
						field.name = sf.Name
					}
					fields = append(fields, field)
					if count[outer.typ] > 1 {
						// Two embeddings of one type at a level annihilate each
						// other's fields below.
						fields = append(fields, field)
					}
					continue
				}
				nextCount[followed]++
				if nextCount[followed] == 1 {
					next = append(next, embedded{typ: followed, index: index})
				}
			}
		}
	}
	slices.SortFunc(fields, func(a, b structField) int {
		if c := strings.Compare(a.name, b.name); c != 0 {
			return c
		}
		if c := cmp.Compare(len(a.index), len(b.index)); c != 0 {
			return c
		}
		if a.tagged != b.tagged {
			if a.tagged {
				return -1
			}
			return 1
		}
		return slices.Compare(a.index, b.index)
	})
	result := fields[:0]
	for start := 0; start < len(fields); {
		end := start + 1
		for end < len(fields) && fields[end].name == fields[start].name {
			end++
		}
		// The first field of a name dominates unless the next one is as
		// shallow and equally tagged, in which case encoding/json drops both.
		if end-start == 1 || len(fields[start].index) != len(fields[start+1].index) ||
			fields[start].tagged != fields[start+1].tagged {
			result = append(result, fields[start])
		}
		start = end
	}
	return result
}

// validTagName is encoding/json's isValidTag.
func validTagName(name string) bool {
	if name == "" {
		return false
	}
	for _, character := range name {
		switch {
		case strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", character):
		case !unicode.IsLetter(character) && !unicode.IsDigit(character):
			return false
		}
	}
	return true
}
