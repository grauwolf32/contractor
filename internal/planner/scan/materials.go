package scan

import (
	"bytes"
	"context"
	"errors"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/stageresult"
)

// Select only exact, already materialized common outputs. Reports remain on
// this Run. Neither the Planner nor this selection interprets provider data.
func (p *execution) selectNormalizedMaterials(ctx context.Context, jobs []planner.ScanJobRecord, result *stageresult.StageContentResult) error {
	slot, selected := p.invocation.Stage.Result.Artifacts["materials"]
	if !selected {
		return nil
	}
	budget := contracts.MaterialBudget{}
	for _, job := range jobs {
		for _, name := range []string{"candidates", "evidence"} {
			ref, exists := job.Outputs[name]
			if !exists {
				continue
			}
			payload, err := p.factory.artifacts.Read(ctx, p.invocation.RunID, ref, planner.MaxScanReportBytes)
			if err != nil {
				return p.scanError("scan_report_write_failed", err)
			}
			if err := budget.Add(contracts.MaterialDescriptor{ScopeKind: "run", ScopeID: p.invocation.RunID, Ref: ref, Digest: contentdigest.Bytes(payload.Data), MediaType: payload.MediaType, SizeBytes: int64(len(payload.Data))}); err != nil {
				return p.scanError("scan_report_invalid", err)
			}
		}
	}
	data, err := strictjson.Canonical(contracts.MaterialSet{Schema: contracts.MaterialSetSchema, Materials: budget.Materials()})
	if err != nil {
		return err
	}
	// The downstream Stage resolves this declared binding. A new attempt may
	// advance it with CAS, while all previous exact result revisions survive.
	target := contracts.ArtifactRef{Namespace: slot.From.Namespace, Name: slot.From.Name}
	payload := artifacts.Payload{MediaType: "application/json", Data: data}
	ref, err := p.factory.artifacts.Resolve(ctx, p.invocation.RunID, target)
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		ref, err = p.factory.artifacts.Create(ctx, p.invocation.RunID, target, payload)
	} else if err == nil {
		var current artifacts.Payload
		current, err = p.factory.artifacts.Read(ctx, p.invocation.RunID, ref, planner.MaxScanArtifactBytes)
		if err == nil && (current.MediaType != payload.MediaType || !bytes.Equal(current.Data, data)) {
			ref, err = p.factory.artifacts.Replace(ctx, p.invocation.RunID, target, payload, *ref.Revision)
		}
	}
	if err != nil {
		return p.scanError("scan_report_write_failed", err)
	}
	result.Artifacts["materials"] = ref
	return nil
}
