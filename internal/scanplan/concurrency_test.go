package scanplan_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func TestPrepareConcurrentCallsPreserveSharedInputs(t *testing.T) {
	data := encoded(t, document(map[string]any{"/x": map[string]any{"get": map[string]any{
		"parameters": []any{map[string]any{"name": "X-ID", "in": "header", "schema": map[string]any{"type": "string"}}},
	}}}))
	options := scanplan.Options{Operations: map[string]scanplan.OperationInput{
		"#/paths/~1x/get": {Parameters: map[string]any{"header:X-ID": "supplied"}},
	}}
	before := encoded(t, options)
	ref := sourceRef()
	expected, err := scanplan.Prepare(data, "application/json", ref, options)
	if err != nil {
		t.Fatal(err)
	}
	want, err := contracts.MarshalHTTPRequestSet(expected)
	if err != nil {
		t.Fatal(err)
	}
	errors := make(chan error, 8)
	for range 8 {
		go func() {
			result, err := scanplan.Prepare(data, "application/json", ref, options)
			if err == nil {
				var wire []byte
				wire, err = contracts.MarshalHTTPRequestSet(result)
				if err == nil && !bytes.Equal(wire, want) {
					err = fmt.Errorf("concurrent output changed")
				}
			}
			errors <- err
		}()
	}
	for range 8 {
		if err := <-errors; err != nil {
			t.Fatal(err)
		}
	}
	if !bytes.Equal(before, encoded(t, options)) {
		t.Fatal("caller options changed")
	}
	*ref.Revision = "later-revision"
	if *expected.Source.Artifact.Revision != "source-1" {
		t.Fatal("retained source revision aliases caller storage")
	}
}
