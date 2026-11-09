package auditdomain

import (
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

const WebTargetSchema = "contractor.audit.web-target.v1"

// PrepareWebTarget validates the target and ordinary filter parameters before
// traffic. Engagement owns authentication and all HTTP request limits.
func PrepareWebTarget(source []byte, parameters map[string]string) (ScanTask, scanplan.PreparedAuditOperation, error) {
	var target struct {
		Schema string `json:"schema" strictjson:"required"`
		URL    string `json:"url" strictjson:"required"`
	}
	arguments := map[string]any{"url": "https://target.invalid", "template_ids": parameters["template_ids"], "tags": parameters["tags"], "severity": parameters["severity"]}
	if len(source) > scanplan.MaxOptionsBytes || strictjson.DecodeNonNull(source, &target) != nil || target.Schema != WebTargetSchema {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "web_scan.target")
	}
	arguments["url"] = target.URL
	if scan.ValidateNucleiArguments(arguments) != nil {
		return ScanTask{}, scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "web_scan.selection")
	}
	selection, _ := strictjson.Canonical(parameters)
	identity, _ := strictjson.Canonical(struct {
		Source    string `json:"source"`
		Selection string `json:"selection"`
	}{DigestBytes(source), DigestBytes(selection)})
	task := ScanTask{Scanner: "nuclei", Operation: "target", SelectionDigest: DigestBytes(selection), PreparationDigest: DigestBytes(identity), Runnable: true, TargetURL: target.URL, TestParameters: []string{}, Gaps: []string{"url_template_scan_only"}}
	return task, scanplan.PreparedAuditOperation{Operation: task.Operation, PreparationDigest: task.PreparationDigest, Runnable: true, Target: &scanplan.OperationTarget{URL: task.TargetURL}}, nil
}
