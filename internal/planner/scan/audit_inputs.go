package scan

import (
	"context"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

type auditScanInputs struct {
	task     auditdomain.ItemTask
	manifest auditdomain.ExecutionManifest
	prepared scanplan.PreparedAuditOperation
}

func (p *execution) loadAuditInputs(ctx context.Context) error {
	task, pkg, err := p.readAuditTask(ctx)
	if err != nil {
		return err
	}
	manifest, err := p.readAuditManifest(ctx, task, pkg)
	if err != nil {
		return err
	}
	source, err := p.readAuditInput(ctx, p.invocation.Stage.ScanPlan.InputArtifact, scanplan.MaxSourceBytes)
	if err != nil {
		return scanError("audit_scan_source_unavailable", err)
	}
	settings, err := p.readAuditInput(ctx, p.invocation.Stage.AuditScan.SettingsArtifact, scanplan.MaxOptionsBytes)
	if err != nil {
		return scanError("audit_scan_settings_unavailable", err)
	}
	if source.MediaType != task.SourceMediaType || settings.MediaType != auditdomain.JSONMediaType {
		return scanError("audit_scan_input_invalid", nil)
	}
	prepared, err := auditdomain.PrepareOpenAPIScanTask(task, source.Data, settings.Data)
	if err != nil {
		return scanError("audit_scan_preparation_invalid", err)
	}
	p.audit = &auditScanInputs{task: task, manifest: manifest, prepared: prepared}
	return nil
}

func (p *execution) readAuditInput(ctx context.Context, slot string, limit int) (artifacts.Payload, error) {
	ref := p.invocation.Context.Artifacts[slot]
	if ref == nil {
		return artifacts.Payload{}, scanError("audit_scan_input_missing", nil)
	}
	return p.factory.artifacts.Read(ctx, p.invocation.RunID, *ref, limit)
}

func (p *execution) readAuditTask(ctx context.Context) (auditdomain.ItemTask, *auditdomain.Package, error) {
	emptyTask := auditdomain.ItemTask{}
	var emptyPackage *auditdomain.Package
	taskPayload, err := p.readAuditInput(ctx, p.invocation.Stage.AuditScan.TaskArtifact, planner.MaxScanArtifactBytes)
	if err != nil {
		return emptyTask, emptyPackage, scanError("audit_scan_task_unavailable", err)
	}
	if taskPayload.MediaType != auditdomain.PackageMediaType {
		return emptyTask, emptyPackage, scanError("audit_scan_task_invalid", nil)
	}
	pkg, err := auditdomain.ValidatePackage(taskPayload.Data)
	if err != nil || pkg.Manifest.Kind != auditdomain.PackageKindTask || len(pkg.Members()) != 1 {
		return emptyTask, emptyPackage, scanError("audit_scan_task_invalid", err)
	}
	member, ok := pkg.MemberByID("task-document")
	if !ok || member.Metadata().MediaType != "application/json" || member.Metadata().Path != "task.json" {
		return emptyTask, emptyPackage, scanError("audit_scan_task_invalid", nil)
	}
	task, err := auditdomain.DecodeItemTask(member.Data())
	if err != nil || task.Scan == nil || task.Scan.Scanner != p.invocation.Stage.AuditScan.Scanner {
		return emptyTask, emptyPackage, scanError("audit_scan_task_invalid", err)
	}
	return task, pkg, nil
}

func (p *execution) readAuditManifest(ctx context.Context, task auditdomain.ItemTask, pkg *auditdomain.Package) (auditdomain.ExecutionManifest, error) {
	empty := auditdomain.ExecutionManifest{}
	manifestPayload, err := p.readAuditInput(ctx, p.invocation.Stage.AuditScan.ManifestArtifact, planner.MaxScanArtifactBytes)
	if err != nil {
		return empty, scanError("audit_scan_manifest_unavailable", err)
	}
	manifest, err := auditdomain.DecodeExecutionManifest(manifestPayload.Data)
	if err != nil || manifestPayload.MediaType != "application/json" || auditdomain.ValidateDispatchExecutionManifest(manifest) != nil || len(manifest.Items) != 1 {
		return empty, scanError("audit_scan_manifest_invalid", err)
	}
	assignment := manifest.Items[0]
	if assignment.ItemKey != task.ItemKey || assignment.SubjectKey != task.SubjectKey || assignment.TaskPackageID != pkg.Manifest.PackageID || assignment.TaskPackageDigest != pkg.Digest {
		return empty, scanError("audit_scan_assignment_invalid", nil)
	}
	// Dispatch manifests use Workflow input aliases, while the task retains the
	// producer's original input identity. Additional inputs may serve other
	// Stages; only these two are consumed by the scanner executor.
	stage := p.invocation.Stage
	sourceName := stage.Context.Artifacts[stage.ScanPlan.InputArtifact].Name
	settingsName := stage.Context.Artifacts[stage.AuditScan.SettingsArtifact].Name
	expected := map[string]auditdomain.ExactInput{
		sourceName:   {Ref: task.SourceRef, Digest: task.SourceContentDigest},
		settingsName: task.Scan.Settings,
	}
	for _, input := range assignment.Inputs {
		pinned, required := expected[input.Name]
		if !required {
			continue
		}
		if input.Digest != pinned.Digest || !sameRef(input.Ref, pinned.Ref) {
			return empty, scanError("audit_scan_assignment_invalid", nil)
		}
		delete(expected, input.Name)
	}
	if len(expected) != 0 {
		return empty, scanError("audit_scan_assignment_invalid", nil)
	}
	return manifest, nil
}

func (p *execution) auditPlanInput(ctx context.Context) (contracts.ArtifactRef, artifacts.Payload, error) {
	payload := artifacts.Payload{}
	var err error
	if p.audit.prepared.RequestSet != nil {
		payload.MediaType = contracts.HTTPRequestSetMediaType
		payload.Data, err = contracts.MarshalHTTPRequestSet(*p.audit.prepared.RequestSet)
	} else {
		payload.MediaType = scanplan.TargetListMediaType
		payload.Data = []byte(p.audit.prepared.Target.URL + "\n")
	}
	if err != nil {
		return contracts.ArtifactRef{}, payload, scanError("audit_scan_preparation_invalid", err)
	}
	ref, err := p.factory.artifacts.Create(ctx, p.invocation.RunID, p.internalTarget("prepared", p.audit.task.Scan.PreparationDigest), payload)
	if err != nil {
		return contracts.ArtifactRef{}, payload, scanError("audit_scan_preparation_write_failed", err)
	}
	return ref, payload, nil
}

func (p *execution) auditHistory(ctx context.Context) ([]planner.ScanAttempt, error) {
	history, err := p.factory.sessions.(planner.AuditScanHistoryReader).ReadAuditScanHistory(ctx, p.invocation.RunID)
	if err != nil {
		return nil, scanError("scan_history_unavailable", err)
	}
	return history, nil
}
