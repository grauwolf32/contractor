// Command openapivalidator validates one evaluator-private OpenAPI artifact.
// It uses the repository's existing validator and never resolves external refs.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"time"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const maxDocument = 1 << 20

func validate(data []byte) error {
	if len(data) > maxDocument {
		return errors.New("OpenAPI document exceeds its bound")
	}
	var object map[string]json.RawMessage
	if err := strictjson.DecodeNonNull(data, &object); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	loader := openapi3.NewLoader()
	loader.Context = ctx
	loader.IsExternalRefsAllowed = false
	document, err := loader.LoadFromData(data)
	if err != nil {
		return err
	}
	return document.Validate(ctx, openapi3.IsOpenAPI31OrLater())
}

func main() {
	data, err := io.ReadAll(io.LimitReader(os.Stdin, maxDocument+1))
	if err != nil || validate(data) != nil {
		// Validation errors can contain artifact content. The scorer needs only
		// a verdict; no private evidence is written to stdout or stderr.
		os.Exit(1)
	}
}
