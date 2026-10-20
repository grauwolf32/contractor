package contracts_test

import (
	"encoding/json"
	"fmt"
	"maps"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// contextRequired lists the members a schema requires in only some of the
// places a shared Go type is used. Validate owns them, so the type cannot tag
// them; every entry names the check.
var contextRequired = map[string]string{
	"contracts.ArtifactRef.revision": "exact refs (stage content, Worker results, Skills, completion " +
		"contracts, read results) require it; ArtifactRef.ValidateExact checks it there",
	"contracts.ResolvedModelPolicy.contextWindowTokens": "a Worker summarizer's policy requires it; " +
		"ResolvedModelPolicy.ValidateForWorkerSummarizer checks it",
	"contracts.ResolvedModelPolicy.maxModelCalls": "a Worker summarizer's policy requires it; " +
		"ResolvedModelPolicy.ValidateForWorkerSummarizer checks it",
	"contracts.ResolvedModelPolicy.maxOutputTokens": "a Worker summarizer's policy requires it; " +
		"ResolvedModelPolicy.ValidateForWorkerSummarizer checks it",
}

// TestSharedRequiredMembersMatchSchemas pins the Go decoding table, the
// strictjson tags strictjson.Decode enforces, to the schemas. Starting from
// every fixture type it walks the schema next to the Go type. Each struct must
// declare every schema member and tag exactly the members every schema context
// requires: `strictjson:"required"`, or `strictjson:"required,nullable"` when
// the schema lets the member be null, which needs a Go type that can hold
// null. A required member is never omitempty, so Go's own encoding keeps it.
// Members required only in some oneOf or anyOf branches, under if/then, or in
// some contexts of a shared type (contextRequired) stay with Validate.
func TestSharedRequiredMembersMatchSchemas(t *testing.T) {
	catalog := loadSchemaCatalog(t)
	index := readFixtureIndex(t)
	pin := requiredPin{catalog: catalog, visited: map[pinKey]bool{}, structs: map[reflect.Type]*structUse{}}
	roots := map[string]string{}
	for _, entries := range []map[string]fixtureIndexEntry{index.Valid, index.Invalid} {
		for _, entry := range entries {
			if entry.Schema != "" {
				roots[entry.Type+" "+entry.Schema] = entry.Type
			}
		}
	}
	keys := make([]string, 0, len(roots))
	for key := range roots {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		typeName := roots[key]
		file, fragment, _ := strings.Cut(strings.TrimPrefix(key, typeName+" "), "#")
		root := schemaNode{file: file, node: catalog[file]}
		if fragment != "" {
			root = catalog.resolve(root, "#"+fragment)
		}
		pin.value(root, fixtureCodecs[typeName].typ, typeName)
	}
	if len(pin.structs) == 0 {
		t.Fatal("no struct was compared with a schema")
	}
	observed := map[string]bool{}
	for _, typ := range sortedStructTypes(pin.structs) {
		use := pin.structs[typ]
		for _, name := range sortedMemberNames(use.members) {
			member := use.members[name]
			if member.tagErr != "" {
				pin.problem("%s.%s: %s", typ, name, member.tagErr)
				continue
			}
			everywhere, somewhere := true, false
			nullEverywhere, nullSomewhere := true, false
			for index, required := range use.required {
				everywhere = everywhere && required[name]
				somewhere = somewhere || required[name]
				nullEverywhere = nullEverywhere && use.nullable[index][name]
				nullSomewhere = nullSomewhere || use.nullable[index][name]
			}
			where := strings.Join(use.paths, ", ")
			if member.required != everywhere {
				pin.problem("%s.%s: tagged required=%v, required in every schema context=%v (%s)",
					typ, name, member.required, everywhere, where)
			}
			if everywhere && nullSomewhere != nullEverywhere {
				pin.problem("%s.%s: the schema lets the required member be null in only some contexts (%s)",
					typ, name, where)
			}
			if member.nullable != nullEverywhere {
				pin.problem("%s.%s: tagged nullable=%v, the schema lets the member be null=%v (%s)",
					typ, name, member.nullable, nullEverywhere, where)
			}
			if member.required && member.omitempty {
				pin.problem("%s.%s: a required member must not be omitempty, or Go would omit it from its own encoding",
					typ, name)
			}
			if somewhere && !everywhere {
				key := typ.String() + "." + name
				observed[key] = true
				if contextRequired[key] == "" {
					pin.problem("%s is required in only some schema contexts (%s); add it to contextRequired with the check that owns it",
						key, where)
				}
			}
		}
	}
	for key := range contextRequired {
		if !observed[key] {
			pin.problem("contextRequired lists %s, which no schema context requires partially", key)
		}
	}
	for _, problem := range pin.problems {
		t.Error(problem)
	}
}

// schemaNode is one schema object and the file that resolves its references.
type schemaNode struct {
	file string
	node map[string]any
}

type schemaCatalog map[string]map[string]any

func loadSchemaCatalog(t *testing.T) schemaCatalog {
	t.Helper()
	paths, err := filepath.Glob(contractstest.Path(t, "api", "v1alpha1", "*.schema.json"))
	if err != nil {
		t.Fatal(err)
	}
	catalog := schemaCatalog{}
	for _, path := range paths {
		var document map[string]any
		if err := json.Unmarshal(contractstest.ReadFile(t, "api", "v1alpha1", filepath.Base(path)), &document); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		catalog[filepath.Base(path)] = document
	}
	return catalog
}

func (c schemaCatalog) resolve(from schemaNode, reference string) schemaNode {
	file, pointer, _ := strings.Cut(reference, "#")
	if file == "" {
		file = from.file
	}
	file = filepath.Base(file)
	node := c[file]
	for _, part := range strings.Split(pointer, "/") {
		if part == "" {
			continue
		}
		node, _ = node[strings.NewReplacer("~1", "/", "~0", "~").Replace(part)].(map[string]any)
	}
	return schemaNode{file: file, node: node}
}

func (c schemaCatalog) children(from schemaNode, keyword string) []schemaNode {
	items, _ := from.node[keyword].([]any)
	result := make([]schemaNode, 0, len(items))
	for _, item := range items {
		if node, ok := item.(map[string]any); ok {
			result = append(result, schemaNode{file: from.file, node: node})
		}
	}
	return result
}

// objectShape is what a schema node says about one JSON value: named
// members with the subschemas describing each, the members it requires
// unconditionally, and its map values and array items.
type objectShape struct {
	properties map[string][]schemaNode
	required   map[string]bool
	additional []schemaNode
	items      []schemaNode
}

func newObjectShape() objectShape {
	return objectShape{properties: map[string][]schemaNode{}, required: map[string]bool{}}
}

func (s *objectShape) merge(other objectShape) {
	for name, nodes := range other.properties {
		s.properties[name] = append(s.properties[name], nodes...)
	}
	for name := range other.required {
		s.required[name] = true
	}
	s.additional = append(s.additional, other.additional...)
	s.items = append(s.items, other.items...)
}

// shape collects n with its $ref and allOf parts. Of oneOf and anyOf it keeps
// the object branches the Go member names can satisfy, preferring an exact
// match, and requires only what all kept branches require.
func (c schemaCatalog) shape(n schemaNode, members map[string]bool) objectShape {
	result := newObjectShape()
	if n.node == nil {
		return result
	}
	if reference, ok := n.node["$ref"].(string); ok {
		result.merge(c.shape(c.resolve(n, reference), members))
	}
	for _, part := range c.children(n, "allOf") {
		result.merge(c.shape(part, members))
	}
	if properties, ok := n.node["properties"].(map[string]any); ok {
		for name, sub := range properties {
			if node, ok := sub.(map[string]any); ok {
				result.properties[name] = append(result.properties[name], schemaNode{file: n.file, node: node})
			}
		}
	}
	if required, ok := n.node["required"].([]any); ok {
		for _, name := range required {
			result.required[name.(string)] = true
		}
	}
	if additional, ok := n.node["additionalProperties"].(map[string]any); ok {
		result.additional = append(result.additional, schemaNode{file: n.file, node: additional})
	}
	if items, ok := n.node["items"].(map[string]any); ok {
		result.items = append(result.items, schemaNode{file: n.file, node: items})
	}
	for _, keyword := range []string{"oneOf", "anyOf"} {
		var objects, exact []objectShape
		for _, branch := range c.children(n, keyword) {
			candidate := c.shape(branch, members)
			if len(candidate.properties) == 0 {
				result.additional = append(result.additional, candidate.additional...)
				result.items = append(result.items, candidate.items...)
				continue
			}
			names := map[string]bool{}
			for name := range candidate.properties {
				names[name] = true
			}
			if !subset(names, members) {
				continue
			}
			objects = append(objects, candidate)
			if len(names) == len(members) {
				exact = append(exact, candidate)
			}
		}
		if len(exact) > 0 {
			objects = exact
		}
		// Only members every kept branch requires are unconditionally required.
		var common map[string]bool
		for index, branch := range objects {
			if index == 0 {
				common = maps.Clone(branch.required)
			} else {
				for name := range common {
					if !branch.required[name] {
						delete(common, name)
					}
				}
			}
			branch.required = nil
			result.merge(branch)
		}
		for name := range common {
			result.required[name] = true
		}
	}
	return result
}

func subset(names, members map[string]bool) bool {
	for name := range names {
		if !members[name] {
			return false
		}
	}
	return true
}

type pinKey struct {
	node uintptr
	typ  reflect.Type
}

type requiredPin struct {
	catalog  schemaCatalog
	visited  map[pinKey]bool
	problems []string
	structs  map[reflect.Type]*structUse
}

// structUse records every schema context that decodes into one Go struct:
// the members each requires and the required members each lets be null.
type structUse struct {
	members  map[string]goMember
	paths    []string
	required []map[string]bool
	nullable []map[string]bool
}

type goMember struct {
	typ       reflect.Type
	required  bool
	nullable  bool
	omitempty bool
	tagErr    string
}

// goMembers lists the JSON members encoding/json decodes into struct t with
// their strictjson tags.
func goMembers(t reflect.Type) map[string]goMember {
	result := map[string]goMember{}
	for index := range t.NumField() {
		field := t.Field(index)
		name, options, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name == "-" || !field.IsExported() && !field.Anonymous {
			continue
		}
		if field.Anonymous && name == "" {
			embedded := field.Type
			if embedded.Kind() == reflect.Pointer {
				embedded = embedded.Elem()
			}
			for promoted, member := range goMembers(embedded) {
				result[promoted] = member
			}
			continue
		}
		if name == "" {
			name = field.Name
		}
		member := goMember{typ: field.Type, omitempty: strings.Contains(","+options+",", ",omitempty,")}
		switch tag := field.Tag.Get("strictjson"); tag {
		case "":
		case "required":
			member.required = true
		case "required,nullable":
			member.required, member.nullable = true, true
		case "nullable":
			member.nullable = true
		default:
			member.tagErr = fmt.Sprintf("strictjson tag %q is not required, required,nullable or nullable", tag)
		}
		result[name] = member
	}
	return result
}

// holdsNull reports whether encoding/json decodes null into t as nil.
func holdsNull(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return true
	default:
		return false
	}
}

// hasRequiredMembers reports whether strictjson rejects null for struct t.
func hasRequiredMembers(t reflect.Type) bool {
	if t.Kind() != reflect.Struct {
		return false
	}
	for _, member := range goMembers(t) {
		if member.required {
			return true
		}
	}
	return false
}

func (p *requiredPin) problem(format string, args ...any) {
	p.problems = append(p.problems, fmt.Sprintf(format, args...))
}

func (p *requiredPin) value(n schemaNode, t reflect.Type, path string) {
	if hasRequiredMembers(t) && p.catalog.allowsNull([]schemaNode{n}) {
		p.problem("%s: the schema allows null, but strictjson rejects null for struct %s with required members",
			path, t)
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	key := pinKey{node: reflect.ValueOf(n.node).Pointer(), typ: t}
	if p.visited[key] {
		return
	}
	p.visited[key] = true
	switch t.Kind() {
	case reflect.Struct:
		members := goMembers(t)
		if len(members) == 0 {
			return
		}
		names := map[string]bool{}
		for name := range members {
			names[name] = true
		}
		shape := p.catalog.shape(n, names)
		if len(shape.properties) == 0 {
			p.problem("%s: Go struct %s has no object schema", path, t)
			return
		}
		use := p.structs[t]
		if use == nil {
			use = &structUse{members: members}
			p.structs[t] = use
		}
		nullable := map[string]bool{}
		for name := range shape.properties {
			if p.catalog.allowsNull(shape.properties[name]) {
				nullable[name] = true
			}
		}
		use.paths = append(use.paths, path)
		use.required = append(use.required, shape.required)
		use.nullable = append(use.nullable, nullable)
		for _, name := range sortedSchemaNames(shape.properties) {
			member, ok := members[name]
			if !ok {
				p.problem("%s: Go struct %s lacks schema member %q", path, t, name)
				continue
			}
			if shape.required[name] && nullable[name] && !holdsNull(member.typ) {
				p.problem("%s/%s: the schema lets the required member be null, but %s cannot hold null",
					path, name, member.typ)
			}
			for _, sub := range shape.properties[name] {
				p.value(sub, member.typ, path+"/"+name)
			}
		}
	case reflect.Map:
		for _, sub := range p.catalog.shape(n, nil).additional {
			p.value(sub, t.Elem(), path+"/*")
		}
	case reflect.Slice, reflect.Array:
		for _, sub := range p.catalog.shape(n, nil).items {
			p.value(sub, t.Elem(), path+"/0")
		}
	}
}

func sortedMemberNames(members map[string]goMember) []string {
	names := make([]string, 0, len(members))
	for name := range members {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func sortedSchemaNames(properties map[string][]schemaNode) []string {
	names := make([]string, 0, len(properties))
	for name := range properties {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func sortedStructTypes(structs map[reflect.Type]*structUse) []reflect.Type {
	types := make([]reflect.Type, 0, len(structs))
	for typ := range structs {
		types = append(types, typ)
	}
	sort.Slice(types, func(i, j int) bool { return types[i].String() < types[j].String() })
	return types
}

// allowsNull reports whether any of nodes accepts a JSON null.
func (c schemaCatalog) allowsNull(nodes []schemaNode) bool {
	for _, n := range nodes {
		if n.node == nil {
			continue
		}
		switch typed := n.node["type"].(type) {
		case string:
			if typed == "null" {
				return true
			}
		case []any:
			for _, item := range typed {
				if item == "null" {
					return true
				}
			}
		}
		if reference, ok := n.node["$ref"].(string); ok && c.allowsNull([]schemaNode{c.resolve(n, reference)}) {
			return true
		}
		if c.allowsNull(c.children(n, "anyOf")) || c.allowsNull(c.children(n, "oneOf")) {
			return true
		}
	}
	return false
}
