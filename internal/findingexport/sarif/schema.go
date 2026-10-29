package sarif

import (
	"bytes"
	_ "embed"
	"errors"
	"sync"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/documentnumber"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

const schemaURI = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
const schemaDigest = "sha256:c3b4bb2d6093897483348925aaa73af03b3e3f4bd4ca38cef26dcb4212a2682e"

//go:embed schema/sarif-schema-2.1.0.json
var schemaBytes []byte

var pinnedSchema = sync.OnceValues(func() (*jsonschema.Schema, error) {
	if contentdigest.Bytes(schemaBytes) != schemaDigest {
		return nil, errors.New("embedded SARIF schema digest is invalid")
	}
	source, err := jsonschema.UnmarshalJSON(bytes.NewReader(schemaBytes))
	if err != nil {
		return nil, err
	}
	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft4)
	compiler.AssertFormat()
	compiler.UseLoader(embeddedOnly{})
	if err := compiler.AddResource(schemaURI, source); err != nil {
		return nil, err
	}
	return compiler.Compile(schemaURI)
})

type embeddedOnly struct{}

func (embeddedOnly) Load(string) (any, error) {
	return nil, errors.New("SARIF validation accepts only the embedded schema")
}

func validateSchema(data []byte) (map[string]any, error) {
	if len(data) > findingexport.MaxBytes {
		return nil, findingexport.ErrLimit
	}
	value, err := strictjson.DecodeBounded(data, strictjson.Bounds{MaxDepth: 96, MaxNodes: 2000000, MaxStringBytes: findingexport.MaxBytes}, documentnumber.Float)
	if err != nil {
		return nil, findingexport.ErrInvalid
	}
	schema, err := pinnedSchema()
	if err != nil {
		return nil, err
	}
	if err := schema.Validate(value); err != nil {
		return nil, findingexport.ErrInvalid
	}
	object, ok := value.(map[string]any)
	if !ok {
		return nil, findingexport.ErrInvalid
	}
	return object, nil
}
