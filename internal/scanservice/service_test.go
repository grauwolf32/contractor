package scanservice

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestSweepRequestRejectsAmbiguousScopeAndMalformedFields(t *testing.T) {
	for _, data := range []string{
		`null`, `{}`, `{"allProjects":false}`, `{"allProjects":true,"workflow":""}`, `{"projectIds":[]}`,
		`{"allProjects":true,"projectIds":["one"]}`, `{"projectIds":["one","one"]}`,
		`{"allProjects":true,"parameters":{"target":"https://unselected.invalid/"}}`,
		`{"allProjects":true,"rulepack":{"namespace":"rules","name":"pack"}}`,
		`{"allProjects":true,"rulepack":null}`, `{"allProjects":true,"parameters":null}`,
		`{"allProjects":true,"allProjects":false}`, `{"AllProjects":true}`, `{"allProjects":true,"unknown":true}`,
	} {
		t.Run(data, func(t *testing.T) {
			var r Request
			err := json.Unmarshal([]byte(data), &r)
			if err == nil {
				_, err = normalize(r)
			}
			if err == nil {
				t.Fatalf("invalid request accepted: %s", data)
			}
		})
	}
}

func TestSweepScopeBoundaryAndCanonicalMembership(t *testing.T) {
	r := Request{ProjectIDs: []string{"second", "first"}}
	normalized, err := normalize(r)
	if err != nil || normalized.ProjectIDs[0] != "first" || normalized.Workflow != "nuclei-web@1" {
		t.Fatalf("normalized=%+v %v", normalized, err)
	}
	if r.ProjectIDs[0] != "second" {
		t.Fatal("caller selection was mutated")
	}
	for _, n := range []int{1000, 1001} {
		ids := make([]string, n)
		for i := range n {
			ids[i] = fmt.Sprintf("project-%04d", i)
		}
		_, err := normalize(Request{ProjectIDs: ids})
		if (err == nil) != (n == 1000) {
			t.Fatalf("%d Projects: %v", n, err)
		}
	}
}
