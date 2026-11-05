package auditcontract

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	PreparedWorkSchema    = "contractor.audit.prepared-work.v1"
	PreparedContextSchema = "contractor.audit.prepared-work-context.v1"
	MaxPreparedWork       = 256
	MaxPreparedMaterials  = MaxPreparedWork + 2
	MaxPreparedWorkBytes  = 1 << 20
)

// Material selects ordinary exact bytes. A descriptor grants no access: the
// caller supplies its admitted source set after checking Artifact ownership.
type Material = contracts.MaterialDescriptor

type PreparedSelection struct {
	Key       string              `json:"key"`
	Workflow  string              `json:"workflow"`
	Objective string              `json:"objective"`
	Inputs    map[string]Material `json:"inputs"`
}

type PreparedWork struct {
	Schema string              `json:"schema"`
	Work   []PreparedSelection `json:"work"`
	Gaps   []string            `json:"gaps"`
}

type PreparedContext struct {
	Schema        string     `json:"schema"`
	Program       Program    `json:"program"`
	ProducerRunID string     `json:"producer_run_id"`
	Materials     []Material `json:"materials"`
}

// DecodePreparedWork validates the complete immutable selection against trusted
// producer/program/material context. The catalog owner additionally checks the
// actual Workflow slots before accepting a task. Rejection returns no work.
func DecodePreparedWork(data, context []byte) (PreparedWork, PreparedContext, error) {
	if len(context) == 0 {
		return PreparedWork{}, PreparedContext{}, errInvalid
	}
	canonical, err := Decode(data, context)
	if err != nil {
		return PreparedWork{}, PreparedContext{}, err
	}
	var work PreparedWork
	var trusted PreparedContext
	if strictjson.Decode(canonical, &work) != nil || work.Schema != PreparedWorkSchema || strictjson.Decode(context, &trusted) != nil {
		return PreparedWork{}, PreparedContext{}, errInvalid
	}
	return work, trusted, nil
}

func validPreparedContext(value map[string]any) bool {
	program, err := strictjson.Canonical(value["program"])
	if err != nil {
		return false
	}
	if _, err := DecodeProgram(program); err != nil {
		return false
	}
	materials := value["materials"].([]any)
	if len(materials) > MaxPreparedMaterials {
		return false
	}
	seen := map[string]bool{}
	for _, raw := range materials {
		material := raw.(map[string]any)
		if material["scope_kind"] != "run" || material["scope_id"] != value["producer_run_id"] {
			return false
		}
		key, err := strictjson.Canonical(material["ref"])
		if err != nil || seen[string(key)] {
			return false
		}
		seen[string(key)] = true
	}
	return true
}

func validPreparedWork(value, trusted map[string]any) bool {
	work := value["work"].([]any)
	if len(work) > MaxPreparedWork {
		return false
	}
	seen := map[string]bool{}
	allowed := map[string]bool{}
	if trusted != nil {
		for _, material := range trusted["materials"].([]any) {
			key, err := strictjson.Canonical(material)
			if err != nil {
				return false
			}
			allowed[string(key)] = true
		}
	}
	for _, raw := range work {
		item := raw.(map[string]any)
		key := item["key"].(string)
		if seen[key] {
			return false
		}
		seen[key] = true
		if trusted == nil {
			continue
		}
		program := trusted["program"].(map[string]any)
		binding, exists := program["workflows"].(map[string]any)[item["workflow"].(string)].(map[string]any)
		if !exists {
			return false
		}
		for slot, selection := range item["inputs"].(map[string]any) {
			mapping, exists := binding["inputs"].(map[string]any)[slot].(map[string]any)
			if !exists || mapping["source"] != "audit-input" && mapping["source"] != "task-input" {
				return false
			}
			key, err := strictjson.Canonical(selection)
			if err != nil || !allowed[string(key)] {
				return false
			}
		}
	}
	return true
}
