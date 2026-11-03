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
	task      auditdomain.ScanTask
	prepared  scanplan.PreparedAuditOperation
	selection *auditdomain.WebScanSettings
}

func (p *execution) loadAuditInputs(ctx context.Context) error {
	source, err := p.readAuditInput(ctx, p.invocation.Stage.ScanPlan.InputArtifact, scanplan.MaxSourceBytes)
	if err != nil {
		return p.scanError("audit_scan_source_unavailable", err)
	}
	settings, err := p.readAuditInput(ctx, p.invocation.Stage.ScanPreparation.SettingsArtifact, scanplan.MaxOptionsBytes)
	if err != nil {
		return p.scanError("audit_scan_settings_unavailable", err)
	}
	if settings.MediaType != "application/json" {
		return p.scanError("audit_scan_input_invalid", nil)
	}
	var task auditdomain.ScanTask
	var prepared scanplan.PreparedAuditOperation
	var selection *auditdomain.WebScanSettings
	config := p.invocation.Stage.ScanPreparation
	if config.Kind == "web-target" {
		if source.MediaType != "application/json" {
			return p.scanError("audit_scan_input_invalid", nil)
		}
		ref := p.invocation.Context.Artifacts[config.SettingsArtifact]
		var value auditdomain.WebScanSettings
		task, prepared, value, err = auditdomain.PrepareWebTarget(source.Data, settings.Data, auditdomain.ExactInput{Name: config.SettingsArtifact, Ref: ref.Clone(), Digest: auditdomain.DigestBytes(settings.Data)})
		selection = &value
	} else {
		operation, readErr := p.readAuditInput(ctx, config.OperationArtifact, auditdomain.MaximumDocumentBytes)
		if readErr != nil {
			return p.scanError("audit_scan_operation_unavailable", readErr)
		}
		if operation.MediaType != "application/json" {
			return p.scanError("audit_scan_input_invalid", nil)
		}
		task, prepared, err = auditdomain.PrepareOpenAPIScanOperation(operation.Data, source.Data, settings.Data, source.MediaType)
	}
	if err != nil || task.Scanner != config.Scanner {
		return p.scanError("audit_scan_preparation_invalid", err)
	}
	p.audit = &auditScanInputs{task: task, prepared: prepared, selection: selection}
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
