package scanplan_test

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

func ExamplePrepare() {
	// The caller reads this content through its authorized exact ArtifactRef.
	source := []byte(`{"openapi":"3.0.4","servers":[{"url":"https://api.example.test"}],"paths":{"/pets/{id}":{"get":{"parameters":[{"name":"id","in":"path","required":true,"schema":{"type":"integer"},"example":7}]}}}}`)
	revision := "openapi-1"
	ref := contracts.ArtifactRef{Namespace: "inputs", Name: "openapi", Revision: &revision}
	set, err := scanplan.Prepare(source, "application/json", ref, scanplan.Options{})
	if err != nil {
		panic(err)
	}
	artifact, err := contracts.MarshalHTTPRequestSet(set)
	if err != nil {
		panic(err)
	}
	// Persist artifact using contracts.HTTPRequestSetMediaType. The strict reader
	// verifies identities, ordering and coverage before a later planner uses it.
	decoded, err := contracts.DecodeHTTPRequestSet(artifact)
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded.Requests[0].Request.URL)
	fmt.Println(decoded.Coverage.Prepared, decoded.Coverage.Complete)
	// Output:
	// https://api.example.test/pets/7
	// 1 true
}
