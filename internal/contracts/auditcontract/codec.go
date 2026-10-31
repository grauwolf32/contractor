// Package auditcontract validates the common Audit document boundary. Offline
// validation checks shape and supplied context; it never grants execution authority.
package auditcontract

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"sync"
	"unicode/utf8"

	auditschema "github.com/grauwolf32/contractor/api/audit/v1"
	wireschema "github.com/grauwolf32/contractor/api/v1alpha1"
	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
	"strings"
)

// MaximumDocumentBytes retains the existing Audit JSON-document ceiling.
// Hypotheses, programs and mutation slices have tighter per-kind bounds.
const MaximumDocumentBytes = 8 << 20

var schema = sync.OnceValues(func() (*jsonschema.Schema, error) {
	c := jsonschema.NewCompiler()
	c.UseLoader(localSchemas{})
	var common any
	if err := strictjson.Decode(wireschema.Common, &common); err != nil {
		return nil, err
	}
	if err := c.AddResource("https://contractor.local/schema/v1alpha1/common.schema.json", common); err != nil {
		return nil, err
	}
	var document any
	if err := strictjson.Decode(auditschema.Document, &document); err != nil {
		return nil, err
	}
	if err := c.AddResource("urn:contractor:audit:v1", document); err != nil {
		return nil, err
	}
	return c.Compile("urn:contractor:audit:v1")
})

type localSchemas struct{}

func (localSchemas) Load(string) (any, error) { return nil, errors.New("schema is not embedded") }

var errInvalid = apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid")

// Decode validates a strict, bounded document. For conclusions/results, context
// must be the caller's exact trusted context, never a field of model output.
// Conclusions receive system-derived provenance; results must match it exactly.
// Returned bytes are canonical ordinary JSON, with no storage or execution effects.
func Decode(data, context []byte) ([]byte, error) {
	value, err := decode(data)
	if err != nil {
		return nil, err
	}
	kind := value["schema"]
	if len(context) > 0 && kind != "contractor.audit.result.v1" && kind != "contractor.audit.conclusions.v1" && kind != "contractor.audit.proposals.v1" {
		return nil, errInvalid
	}
	switch kind {
	case "contractor.audit.program.v1":
		if !validProgram(value) {
			return nil, errInvalid
		}
	case "contractor.audit.proposal-context.v1":
		if !validProposalContext(value) {
			return nil, errInvalid
		}
	case "contractor.audit.proposals.v1":
		trusted, contextErr := decode(context)
		if contextErr != nil || trusted["schema"] != "contractor.audit.proposal-context.v1" || !validProposalContext(trusted) {
			return nil, errInvalid
		}
		program := trusted["program"].(map[string]any)
		if !validProposals(value, program, trusted["materials"].(map[string]any)) {
			return nil, errInvalid
		}
	case "contractor.audit.context.v1":
		if !validContext(value) {
			return nil, errInvalid
		}
	case "contractor.audit.result.v1", "contractor.audit.conclusions.v1":
		trusted, contextErr := decode(context)
		if contextErr != nil || trusted["schema"] != "contractor.audit.context.v1" || !validContext(trusted) || !assemble(value, trusted) {
			return nil, errInvalid
		}
	}
	result, err := strictjson.Canonical(value)
	if err != nil || len(result) > documentLimit(value["schema"].(string)) {
		return nil, errInvalid
	}
	return result, nil
}

func documentLimit(kind string) int {
	switch kind {
	case "contractor.audit.hypothesis.v1":
		return 32768
	case "contractor.audit.mutation.v1":
		return 1 << 20
	case "contractor.audit.program.v1":
		return 4 << 20
	default:
		return MaximumDocumentBytes
	}
}

func decode(data []byte) (map[string]any, error) {
	if len(data) == 0 || len(data) > MaximumDocumentBytes || !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) {
		return nil, errInvalid
	}
	raw, parseErr := strictjson.DecodeBounded(data, strictjson.Bounds{
		MaxDepth: 64, MaxNodes: 250000, MaxStringBytes: 65536,
	}, func(s string) (float64, bool) {
		n, err := strconv.ParseInt(s, 10, 64)
		return float64(n), err == nil && n >= -contracts.MaxExactInteger && n <= contracts.MaxExactInteger
	})
	value, ok := raw.(map[string]any)
	if parseErr != nil || !ok {
		return nil, errInvalid
	}
	validator, err := schema()
	if err != nil {
		return nil, err
	}
	if validator.Validate(value) != nil || !validBytes(value) || !validSelectors(value) || !validHypothesisSizes(value) {
		return nil, errInvalid
	}
	if len(data) > documentLimit(value["schema"].(string)) {
		return nil, errInvalid
	}

	if value["schema"] == "contractor.audit.mutation.v1" {
		seen := map[string]bool{}
		for _, record := range value["records"].([]any) {
			item := record.(map[string]any)
			identity := fmt.Sprint(item["schema"], "/", item["id"], "/", item["revision"], "/", item["task_id"])
			if seen[identity] {
				return nil, errInvalid
			}
			seen[identity] = true
			encoded, e := strictjson.Canonical(record)
			if e != nil {
				return nil, errInvalid
			}
			if _, e = decode(encoded); e != nil {
				return nil, errInvalid
			}
		}
	}
	return value, nil
}

func validHypothesisSizes(value map[string]any) bool {
	if value["schema"] != "contractor.audit.context.v1" {
		return true
	}
	for _, raw := range value["items"].([]any) {
		if hypothesis, ok := raw.(map[string]any)["hypothesis"]; ok {
			data, err := strictjson.Canonical(hypothesis)
			if err != nil || len(data) > 32768 {
				return false
			}
		}
	}
	return true
}

func checkingBinding(binding map[string]any) bool {
	if _, ok := binding["outputs"].(map[string]any)["result"]; !ok {
		return false
	}
	for _, raw := range binding["inputs"].(map[string]any) {
		if raw.(map[string]any)["source"] == "task-context" {
			return true
		}
	}
	return false
}

func validSelectors(value map[string]any) bool {
	valid := func(raw string) bool {
		id, version, found := strings.Cut(raw, "@")
		return found && contracts.ValidConfigID(id) && contracts.ValidConfigVersion(version)
	}
	switch value["schema"] {
	case "contractor.audit.context.v1":
		return valid(value["workflow_ref"].(string))
	case "contractor.audit.program.v1":
		for _, raw := range value["workflows"].(map[string]any) {
			if !valid(raw.(map[string]any)["ref"].(string)) {
				return false
			}
		}
	}
	return true
}

func validBytes(value any) bool {
	switch v := value.(type) {
	case string:
		return len(v) <= 65536 && !strings.ContainsRune(v, '\x00')
	case []any:
		for _, item := range v {
			if !validBytes(item) {
				return false
			}
		}
	case map[string]any:
		for key, item := range v {
			if key == "parameters" {
				if parameters, ok := item.(map[string]any); ok {
					for _, raw := range parameters {
						if text, ok := raw.(string); ok && len(text) > 4096 {
							return false
						}
					}
				}
			}
			if key == "limitations" {
				if values, ok := item.([]any); ok {
					for _, raw := range values {
						if text, ok := raw.(string); ok && len(text) > 512 {
							return false
						}
					}
				}
			}
			if !validBytes(key) || !validBytes(item) {
				return false
			}
		}
	}
	return true
}

func validProgram(v map[string]any) bool {
	workflows := v["workflows"].(map[string]any)
	inputs := v["inputs"].(map[string]any)
	hypotheses := v["hypotheses"].(map[string]any)
	limits := v["limits"].(map[string]any)
	if limits["batch_size"].(float64) > limits["max_tasks_per_iteration"].(float64) ||
		limits["max_tasks_per_iteration"].(float64) > limits["max_tasks"].(float64) ||
		limits["max_submitted_runs"].(float64) < math.Ceil(limits["max_tasks"].(float64)/limits["batch_size"].(float64)) ||
		limits["max_hypothesis_revisions"].(float64) < limits["max_hypotheses"].(float64) ||
		len(hypotheses) > int(limits["max_hypotheses"].(float64)) ||
		len(v["initial_work"].([]any)) > int(limits["max_tasks_per_iteration"].(float64)) {
		return false
	}
	for _, raw := range workflows {
		binding := raw.(map[string]any)
		for _, item := range binding["inputs"].(map[string]any) {
			source := item.(map[string]any)
			if source["source"] == "audit-input" {
				if _, ok := inputs[source["name"].(string)]; !ok {
					return false
				}
			}
		}
	}
	seen := map[string]bool{}
	for _, raw := range v["initial_work"].([]any) {
		task := raw.(map[string]any)
		key := task["key"].(string)
		if seen[key] {
			return false
		}
		seen[key] = true
		if _, ok := workflows[task["workflow"].(string)]; !ok {
			return false
		}
		if !validInputSelections(task, workflows[task["workflow"].(string)].(map[string]any), inputs, nil) {
			return false
		}
		if h, ok := task["hypothesis"]; ok {
			if !checkingBinding(workflows[task["workflow"].(string)].(map[string]any)) {
				return false
			}
			if _, ok = hypotheses[h.(string)]; !ok {
				return false
			}
		}
	}
	if c, ok := v["continuation"].(map[string]any); ok {
		if _, ok := workflows[c["workflow"].(string)]; !ok {
			return false
		}
	}
	return true
}

func validProposalContext(value map[string]any) bool {
	data, err := strictjson.Canonical(value["program"])
	if err != nil {
		return false
	}
	_, err = Decode(data, nil)
	return err == nil
}

func validInputSelections(task, binding, inputs, materials map[string]any) bool {
	selected, ok := task["inputs"].(map[string]any)
	if !ok {
		return true
	}
	for slot, raw := range selected {
		mapping, ok := binding["inputs"].(map[string]any)[slot].(map[string]any)
		if !ok || mapping["source"] != "audit-input" {
			return false
		}
		name := raw.(string)
		if materials == nil {
			selected, ok := inputs[name].(map[string]any)
			if !ok {
				return false
			}
			target := inputs[mapping["name"].(string)].(map[string]any)
			allowed := []string{}
			for _, raw := range target["media_types"].([]any) {
				allowed = append(allowed, raw.(string))
			}
			for _, raw := range selected["media_types"].([]any) {
				if !contracts.AcceptsMediaType(allowed, raw.(string)) {
					return false
				}
			}
			continue
		}
		material, ok := materials[name].(map[string]any)
		if !ok {
			return false
		}
		declaration := inputs[mapping["name"].(string)].(map[string]any)
		allowed := []string{}
		for _, media := range declaration["media_types"].([]any) {
			allowed = append(allowed, media.(string))
		}
		if !contracts.AcceptsMediaType(allowed, material["media_type"].(string)) {
			return false
		}
	}
	return true
}

func validProposals(v, program, materials map[string]any) bool {
	seeds := v["hypotheses"].(map[string]any)
	work := v["work"].([]any)
	if len(seeds)+len(work) == 0 || len(seeds)+len(work) > 128 {
		return false
	}
	aliases := map[string]bool{}
	for key := range program["hypotheses"].(map[string]any) {
		aliases[key] = true
	}
	for key := range seeds {
		if aliases[key] {
			return false
		}
		aliases[key] = true
	}
	seen := map[string]bool{}
	for _, raw := range work {
		task := raw.(map[string]any)
		key := task["key"].(string)
		if seen[key] {
			return false
		}
		seen[key] = true
		if _, ok := program["workflows"].(map[string]any)[task["workflow"].(string)]; !ok {
			return false
		}
		if !validInputSelections(task, program["workflows"].(map[string]any)[task["workflow"].(string)].(map[string]any), program["inputs"].(map[string]any), materials) {
			return false
		}
		if key, ok := task["hypothesis"].(string); ok {
			if !aliases[key] || !checkingBinding(program["workflows"].(map[string]any)[task["workflow"].(string)].(map[string]any)) {
				return false
			}
		}
	}
	return true
}

func validContext(v map[string]any) bool {
	seen := map[string]bool{}
	tasks := map[string]bool{}
	for _, raw := range v["items"].([]any) {
		item := raw.(map[string]any)
		key := item["item_key"].(string)
		task := item["task_id"].(string)
		if seen[key] || tasks[task] {
			return false
		}
		seen[key] = true
		tasks[task] = true
	}
	return true
}

func assemble(v, context map[string]any) bool {
	expected := context["items"].([]any)
	results := v["items"].([]any)
	if len(expected) != len(results) {
		return false
	}
	authored := v["schema"] == "contractor.audit.conclusions.v1"
	members := map[string]map[string]any{}
	for _, raw := range expected {
		item := raw.(map[string]any)
		members[item["item_key"].(string)] = item
	}
	for _, raw := range results {
		item := raw.(map[string]any)
		key := item["item_key"].(string)
		member, ok := members[key]
		if !ok {
			return false
		}
		delete(members, key)
		hypothesis, checking := member["hypothesis"]
		verdict, hasVerdict := item["verdict"]
		if checking != hasVerdict {
			return false
		}
		if checking && verdict != "inconclusive" && (len(item["tested_conditions"].([]any)) == 0 || len(item["evidence"].([]any)) == 0) {
			return false
		}
		if verdict == "inconclusive" && len(item["limitations"].([]any)) == 0 {
			return false
		}
		for _, evidence := range item["evidence"].([]any) {
			found := false
			for _, allowed := range member["evidence"].([]any) {
				if reflect.DeepEqual(evidence, allowed) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
		origin := map[string]any{"audit_id": context["audit_id"], "run_id": context["run_id"], "task_id": member["task_id"]}
		if checking {
			snapshot := hypothesis.(map[string]any)
			origin["hypothesis"] = map[string]any{"id": snapshot["id"], "revision": snapshot["revision"]}
		}
		receipts, hasReceipts := member["finding_receipts"]
		if authored {
			if hasReceipts {
				item["finding_receipts"] = receipts
			}
			item["origin"] = origin
		} else if !reflect.DeepEqual(item["origin"], origin) || !reflect.DeepEqual(item["finding_receipts"], receipts) {
			return false
		}
	}
	v["schema"] = "contractor.audit.result.v1"
	return len(members) == 0
}
