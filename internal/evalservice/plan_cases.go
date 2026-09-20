package evalservice

import (
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/evaldomain"
)

type planBuilder struct {
	bundle    PlanBundle
	draft     evaldomain.Draft
	dataset   evaldomain.DatasetInput
	preflight map[string]Preflight
}

func (b *planBuilder) add(path, kind string, value any) (documentRef, error) {
	raw, err := jsonBytes(value)
	if err != nil {
		return documentRef{}, err
	}
	doc, err := evaldomain.Freeze(kind, raw)
	if err != nil {
		return documentRef{}, err
	}
	b.bundle.Resources = append(b.bundle.Resources, Resource{Path: path, Kind: kind, Document: doc})
	return documentRef{Resource: path, SHA256: doc.Digest()}, nil
}

func (b *planBuilder) buildChecks() ([]portableCheck, map[string]blobRef, error) {
	checks := make([]portableCheck, 0, len(b.draft.Checks))
	truth := map[string]blobRef{}
	private := map[string]evaldomain.PrivateCheck{}
	for _, check := range b.dataset.PrivateChecks {
		private[check.ID+"@"+check.Revision] = check
	}
	for i := range b.draft.Checks {
		check := &b.draft.Checks[i]
		// A caller-supplied hash never selects an arbitrary implementation.
		if check.Evaluator != "human-review@1" {
			return nil, nil, evaldomain.Failure("eval_not_ready")
		}
		rubric, ok := private[check.ID+"@"+check.RubricRevision]
		if !ok {
			return nil, nil, evaldomain.Failure("eval_not_ready")
		}
		if check.ImplementationSHA256 != "" && check.ImplementationSHA256 != HumanPolicySHA256() {
			return nil, nil, evaldomain.Failure("eval_pin_mismatch")
		}
		check.ImplementationSHA256 = HumanPolicySHA256()
		raw, err := jsonBytes(rubric)
		if err != nil {
			return nil, nil, err
		}
		doc, err := evaldomain.Freeze("PrivateCheck", raw)
		if err != nil {
			return nil, nil, err
		}
		path := "private/" + check.ID + ".json"
		b.bundle.Private[path] = doc
		truth[check.ID] = blobRef{Resource: path, SHA256: doc.Digest(), MediaType: "application/json", SizeBytes: int64(len(raw))}
		params := check.Parameters
		if params == nil {
			params = map[string]string{}
		}
		checks = append(checks, portableCheck{
			ID:                   check.ID,
			Scorer:               check.Evaluator,
			ImplementationSHA256: check.ImplementationSHA256,
			Parameters:           params,
			GroundTruthRole:      &check.ID,
			Required:             check.Required,
			AllowNotApplicable:   check.AllowNotApplicable,
		})
	}
	return checks, truth, nil
}

func (b *planBuilder) buildCases(truth map[string]blobRef) ([]evaldomain.Case, []documentRef, error) {
	byID := map[string]evaldomain.Case{}
	for _, c := range b.dataset.Cases {
		byID[c.ID] = c
	}
	cases := make([]evaldomain.Case, 0, len(b.draft.CaseIDs))
	refs := make([]documentRef, 0, len(b.draft.CaseIDs))
	for _, id := range b.draft.CaseIDs {
		c, ok := byID[id]
		if !ok {
			return nil, nil, evaldomain.Failure("eval_not_found")
		}
		inputs := map[string]blobRef{}
		for role, artifact := range c.Inputs {
			path := "inputs/" + id + "/" + role
			b.bundle.Inputs[path] = artifact
			inputs[role] = blobRef{Resource: path, SHA256: artifact.SHA256, MediaType: artifact.MediaType, SizeBytes: artifact.SizeBytes}
		}
		outputs := map[string]portableOutput{}
		for role, output := range c.Outputs {
			outputs[role] = portableOutput{MediaTypes: output.MediaTypes, Required: output.Required}
		}
		document := portableCase{SchemaVersion: portableCaseSchema, ID: id, Task: c.Task, Inputs: inputs, Requires: c.Requires, Outputs: outputs,
			Evaluation: caseEvaluation{GroundTruth: truth, Assertions: map[string]json.RawMessage{}},
			Provenance: caseProvenance{
				Sources:  []documentRef{},
				Datasets: []datasetProvenance{{ID: b.dataset.DatasetID, Revision: b.draft.Dataset.Revision}},
			},
		}
		ref, err := b.add("cases/"+id+".json", portableCaseSchema, document)
		if err != nil {
			return nil, nil, err
		}
		cases = append(cases, c)
		refs = append(refs, ref)
	}
	return cases, refs, nil
}
