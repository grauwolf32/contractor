package evaldomain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"strings"
	"sync"

	evalschema "github.com/grauwolf32/contractor/api/evals/v1"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

var schemaCatalog = sync.OnceValues(compileSchemas)

type noRemoteSchemas struct{}

func (noRemoteSchemas) Load(string) (any, error) {
	return nil, fmt.Errorf("schema is not in the embedded catalog")
}

func compileSchemas() (map[string]*jsonschema.Schema, error) {
	c := jsonschema.NewCompiler()
	c.DefaultDraft(jsonschema.Draft2020)
	c.AssertFormat()
	c.UseLoader(noRemoteSchemas{})
	ids := map[string]string{}
	err := fs.WalkDir(evalschema.Files, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(path, ".schema.json") {
			return nil
		}
		data, err := evalschema.Files.ReadFile(path)
		if err != nil {
			return err
		}
		value, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
		if err != nil {
			return err
		}
		object := value.(map[string]any)
		id := object["$id"].(string)
		if err := c.AddResource(id, value); err != nil {
			return err
		}
		if id == evalschema.ManagedID {
			for name := range object["$defs"].(map[string]any) {
				ids[name] = id + "#/$defs/" + name
			}
		} else if id != "urn:playground:portable:common" {
			ids[strings.Replace(id, "urn:playground:portable:", "playground.", 1)] = id
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := make(map[string]*jsonschema.Schema, len(ids))
	for name, id := range ids {
		schema, err := c.Compile(id)
		if err != nil {
			return nil, err
		}
		out[name] = schema
	}
	return out, nil
}

// Validate checks closed JSON schemas and cross-field semantic invariants.
// Authorization and catalog resolution belong to the service, not this codec.
func Validate(kind string, data []byte) error {
	value, err := StrictJSON(data)
	if err != nil {
		return err
	}
	catalog, err := schemaCatalog()
	if err != nil {
		return fmt.Errorf("compile embedded eval schemas: %w", err)
	}
	schema, ok := catalog[kind]
	if !ok {
		return Failure("eval_invalid")
	}
	if err = schema.Validate(value); err != nil {
		return Failure("eval_invalid")
	}
	if strings.HasPrefix(kind, "playground.") {
		if err := validatePortableResources(value); err != nil {
			return err
		}
	}
	if object, ok := value.(map[string]any); ok {
		return validateSemantics(kind, object)
	}
	return nil
}

// DecodeInto validates the original bytes first. It does not canonicalize a
// retained portable document or guess an HTTP resource's effective owner.
func DecodeInto(kind string, data []byte, target any) error {
	if err := Validate(kind, data); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	d.UseNumber()
	if err := d.Decode(target); err != nil {
		return Failure("eval_invalid")
	}
	return nil
}

// Frozen keeps its private payload unexported to prevent accidental publication.
// Callers must explicitly project public data; encoding/json sees no raw fields.
type Frozen struct {
	kind   string
	raw    []byte
	sha256 string
}

func Freeze(kind string, data []byte) (Frozen, error) {
	if err := Validate(kind, data); err != nil {
		return Frozen{}, err
	}
	return Frozen{kind: kind, raw: bytes.Clone(data), sha256: Digest(data)}, nil
}

func (f Frozen) Kind() string { return f.kind }

func (f Frozen) Digest() string { return f.sha256 }

func (f Frozen) Bytes() []byte { return bytes.Clone(f.raw) }
