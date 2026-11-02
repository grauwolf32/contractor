package auditdomain

import (
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

const WebTargetSchema = "contractor.audit.web-target.v1"
const WebScanSettingsSchema = "contractor.audit.web-scan-settings.v1"

// WebScanSettings is the exact selection retained with a direct-target Audit.
// The shared Nuclei validator owns filter, severity and request-rate grammar.
type WebScanSettings struct {
	Schema      string `json:"schema" strictjson:"required"`
	TemplateIDs string `json:"templateIds" strictjson:"required"`
	Tags        string `json:"tags" strictjson:"required"`
	Severity    string `json:"severity" strictjson:"required"`
	RateLimit   int    `json:"rateLimit" strictjson:"required"`
}

func DecodeWebScanSettings(data []byte) (WebScanSettings, error) {
	value := WebScanSettings{}
	if len(data) > scanplan.MaxOptionsBytes || strictjson.DecodeNonNull(data, &value) != nil || value.Schema != WebScanSettingsSchema ||
		scan.ValidateNucleiArguments(value.Arguments("https://target.invalid")) != nil {
		return WebScanSettings{}, invalid(CodeInventoryInvalid, "web_scan.settings")
	}
	return value, nil
}

func (s WebScanSettings) Arguments(target string) map[string]any {
	return map[string]any{"url": target, "template_ids": s.TemplateIDs, "tags": s.Tags,
		"severity": s.Severity, "rate_limit": s.RateLimit}
}

func webScanTask(source, settings []byte, input ExactInput) (ScanTask, error) {
	input.Ref = input.Ref.Clone()
	var target struct {
		Schema string `json:"schema" strictjson:"required"`
		URL    string `json:"url" strictjson:"required"`
	}
	selection, err := DecodeWebScanSettings(settings)
	if err != nil || len(source) > scanplan.MaxOptionsBytes || strictjson.DecodeNonNull(source, &target) != nil ||
		target.Schema != WebTargetSchema || scan.ValidateNucleiArguments(selection.Arguments(target.URL)) != nil {
		return ScanTask{}, invalid(CodeInventoryInvalid, "web_scan.target")
	}
	identity, err := strictjson.Canonical(struct {
		Source   string `json:"source"`
		Settings string `json:"settings"`
	}{DigestBytes(source), DigestBytes(settings)})
	if err != nil {
		return ScanTask{}, err
	}
	return ScanTask{Scanner: "nuclei", Operation: "target", Settings: input,
		PreparationDigest: DigestBytes(identity), Runnable: true, TargetURL: target.URL,
		TestParameters: []string{}, Gaps: []string{"url_template_scan_only"}}, nil
}

// BuildWebScanInventory creates one direct-target task without OpenAPI, model
// judgement or network activity. The normal Audit baseline pins every input.
func BuildWebScanInventory(source []byte, mediaType string, settings []byte, input ExactInput, options InventoryOptions) (Inventory, error) {
	if mediaType != JSONMediaType || input.Name == options.SourceInputName ||
		validateIdentifier(input.Name, "settings.name") != nil || input.Ref.ValidateExact() != nil || input.Digest != DigestBytes(settings) ||
		(options.ApprovalRequirement != ApprovalNone && options.ApprovalRequirement != ApprovalActiveCheck) {
		return Inventory{}, invalid(CodeInventoryInvalid, "web_scan.inputs")
	}
	task, err := webScanTask(source, settings, input)
	if err != nil {
		return Inventory{}, err
	}
	basis := inventoryBasis{Schema: InventoryBasisSchema, Kind: "web-scans", Subjects: []map[string]any{{"scan": task}}, Gaps: []string{}}
	canonical, err := canonicalJSON(basis)
	if err != nil {
		return Inventory{}, err
	}
	key := openAPIScanKey(DigestBytes(canonical), task.Operation)
	return finishInventory(source, mediaType, basis, []inventorySubject{{itemKey: key, subjectKey: key,
		kind: "web-scan", scan: &task, approval: options.ApprovalRequirement,
		requested: []string{task.CoverageRequirement()}, gaps: task.Gaps}}, options)
}

// PrepareWebScanTask reproduces the retained assignment before any traffic.
func PrepareWebScanTask(task ItemTask, source, settings []byte) (scanplan.PreparedAuditOperation, error) {
	if task.Kind != "web-scan" || validateItemTask(task) != nil || task.Scan == nil ||
		DigestBytes(source) != task.SourceContentDigest || DigestBytes(settings) != task.Scan.Settings.Digest {
		return scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "web_scan.inputs")
	}
	expected, err := webScanTask(source, settings, task.Scan.Settings)
	if err != nil || !sameCanonicalValue(expected, *task.Scan) {
		return scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "web_scan.preparation")
	}
	return scanplan.PreparedAuditOperation{Operation: expected.Operation, PreparationDigest: expected.PreparationDigest,
		Runnable: true, Target: &scanplan.OperationTarget{URL: expected.TargetURL}}, nil
}
