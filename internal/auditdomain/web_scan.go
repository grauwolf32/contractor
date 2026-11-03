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

// PrepareWebTarget validates one ordinary target and exact settings before
// traffic. The same domain preparation owns its identity, target and gaps.
func PrepareWebTarget(source, settings []byte, input ExactInput) (ScanTask, scanplan.PreparedAuditOperation, WebScanSettings, error) {
	if input.Ref.ValidateExact() != nil || input.Digest != DigestBytes(settings) {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, WebScanSettings{}, invalid(CodeInventoryInvalid, "web_scan.settings")
	}
	task, err := webScanTask(source, settings, input)
	if err != nil {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, WebScanSettings{}, err
	}
	selection, err := DecodeWebScanSettings(settings)
	if err != nil {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, WebScanSettings{}, err
	}
	return task, scanplan.PreparedAuditOperation{Operation: task.Operation, PreparationDigest: task.PreparationDigest, Runnable: true, Target: &scanplan.OperationTarget{URL: task.TargetURL}}, selection, nil
}
