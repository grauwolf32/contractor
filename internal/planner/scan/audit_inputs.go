package scan

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	contractscan "github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

type auditScanInputs struct {
	task     auditdomain.ScanTask
	prepared scanplan.PreparedAuditOperation
}

func (p *execution) loadAuditInputs(ctx context.Context) error {
	source, err := p.readAuditInput(ctx, p.invocation.Stage.ScanPlan.InputArtifact, scanplan.MaxSourceBytes)
	if err != nil {
		return p.scanError("audit_scan_source_unavailable", err)
	}
	config := p.invocation.Stage.ScanPreparation
	var values []byte
	if config.ValuesArtifact != "" && p.invocation.Context.Artifacts[config.ValuesArtifact] != nil {
		payload, readErr := p.readAuditInput(ctx, config.ValuesArtifact, scanplan.MaxOptionsBytes)
		if readErr != nil {
			return p.scanError("audit_scan_request_values_unavailable", readErr)
		}
		if payload.MediaType != "application/json" {
			return p.scanError("audit_scan_input_invalid", nil)
		}
		values = payload.Data
	}
	var task auditdomain.ScanTask
	var prepared scanplan.PreparedAuditOperation
	if config.Kind == "web-target" {
		if source.MediaType != "application/json" {
			return p.scanError("audit_scan_input_invalid", nil)
		}
		task, prepared, err = auditdomain.PrepareWebTarget(source.Data, p.invocation.Context.Parameters)
	} else {
		operation, readErr := p.readAuditInput(ctx, config.OperationArtifact, auditdomain.MaximumDocumentBytes)
		if readErr != nil || operation.MediaType != "application/json" {
			return p.scanError("audit_scan_operation_unavailable", readErr)
		}
		task, prepared, err = auditdomain.PrepareOpenAPIScanOperation(operation.Data, source.Data, values, source.MediaType, config.Scanner, p.invocation.Context.Parameters)
	}
	if err != nil || task.Scanner != config.Scanner {
		return p.scanError("audit_scan_preparation_invalid", err)
	}
	p.audit = &auditScanInputs{task: task, prepared: prepared}
	return nil
}

func (p *execution) readAuditInput(ctx context.Context, slot string, limit int) (artifacts.Payload, error) {
	ref := p.invocation.Context.Artifacts[slot]
	if ref == nil {
		return artifacts.Payload{}, p.scanError("audit_scan_input_missing", nil)
	}
	return p.factory.artifacts.Read(ctx, p.invocation.RunID, *ref, limit)
}

func (p *execution) auditPlanInput(ctx context.Context) (contracts.ArtifactRef, artifacts.Payload, error) {
	payload := artifacts.Payload{}
	var err error
	if p.audit.prepared.RequestSet != nil {
		payload.MediaType = contractscan.HTTPRequestSetMediaType
		payload.Data, err = contractscan.MarshalHTTPRequestSet(*p.audit.prepared.RequestSet)
	} else {
		payload.MediaType = scanplan.TargetListMediaType
		payload.Data = []byte(p.audit.prepared.Target.URL + "\n")
	}
	if err != nil {
		return contracts.ArtifactRef{}, payload, p.scanError("audit_scan_preparation_invalid", err)
	}
	ref, err := p.factory.artifacts.Create(ctx, p.invocation.RunID, p.internalTarget("prepared", p.audit.task.PreparationDigest), payload)
	if err != nil {
		return contracts.ArtifactRef{}, payload, p.scanError("audit_scan_preparation_write_failed", err)
	}
	return ref, payload, nil
}

func (p *execution) auditHistory(ctx context.Context) ([]planner.ScanAttempt, error) {
	history, err := p.factory.sessions.(planner.AuditScanHistoryReader).ReadAuditScanHistory(ctx, p.invocation.RunID)
	if err != nil {
		return nil, p.scanError("scan_history_unavailable", err)
	}
	return history, nil
}
