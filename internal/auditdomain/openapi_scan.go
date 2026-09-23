package auditdomain

import (
	"errors"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// OpenAPIScanTask records an assigned scan, separately from operation tracing.
// Concrete request bytes and credentials stay in the exact supplied inputs.
// Re-preparation must match this identity before a scanner can be invoked.
type OpenAPIScanTask struct {
	Scanner           string     `json:"scanner"`
	Operation         string     `json:"operation"`
	Settings          ExactInput `json:"settings"`
	PreparationDigest string     `json:"preparation_digest"`
	Runnable          bool       `json:"runnable"`
	RequestDigest     string     `json:"request_digest,omitempty"`
	TargetURL         string     `json:"target_url,omitempty"`
	TestParameters    []string   `json:"test_parameters"`
	Gaps              []string   `json:"gaps"`
}

func (s OpenAPIScanTask) CoverageRequirement() string {
	if s.Scanner == "nuclei" {
		return "nuclei-url-template-scan"
	}
	return "sqlmap-request-scan"
}

// BuildOpenAPIScanInventory selects only the operations explicitly listed in
// settings. Missing concrete data remain non-runnable items with gaps; a missing
// operation or malformed settings reject the entire inventory atomically.
func BuildOpenAPIScanInventory(source []byte, sourceMediaType string, settingsData []byte, settingsInput ExactInput, options InventoryOptions) (Inventory, error) {
	if options.ApprovalRequirement != ApprovalActiveCheck || settingsInput.Name == options.SourceInputName ||
		validateIdentifier(settingsInput.Name, "settings.name") != nil || settingsInput.Ref.ValidateExact() != nil ||
		settingsInput.Digest != DigestBytes(settingsData) {
		return Inventory{}, invalid(CodeInventoryInvalid, "scan.settings")
	}
	settings, err := scanplan.DecodeAuditScanSettings(settingsData)
	if err != nil {
		return Inventory{}, invalid(CodeInventoryInvalid, "scan.settings")
	}
	basis := inventoryBasis{Schema: InventoryBasisSchema, Kind: "openapi-scans", Subjects: []map[string]any{}, Gaps: []string{}}
	subjects := []inventorySubject{}
	for _, pointer := range settings.Operations() {
		prepared, err := settings.PrepareOperation(source, normalizedMediaType(sourceMediaType), options.SourceRef, pointer)
		if err != nil {
			var failure *scanplan.PreparationError
			if errors.As(err, &failure) {
				return Inventory{}, invalid(CodeInventoryInvalid, "scan."+failure.Code)
			}
			return Inventory{}, invalid(CodeInventoryInvalid, "scan.preparation")
		}
		task := openAPIScanTask(settings, settingsInput, prepared)
		subjects = append(subjects, inventorySubject{kind: "openapi-scan", scan: &task,
			approval: ApprovalActiveCheck, requested: []string{task.CoverageRequirement()}, gaps: task.Gaps})
		basis.Subjects = append(basis.Subjects, map[string]any{"scan": task})
	}
	canonical, err := canonicalJSON(basis)
	if err != nil {
		return Inventory{}, invalid(CodeInventoryInvalid, "scan.inventory")
	}
	canonicalDigest := DigestBytes(canonical)
	for index := range subjects {
		subjects[index].itemKey = openAPIScanKey(canonicalDigest, subjects[index].scan.Operation)
		subjects[index].subjectKey = subjects[index].itemKey
	}
	return finishInventory(source, sourceMediaType, basis, subjects, options)
}

func openAPIScanTask(settings scanplan.AuditScanSettings, input ExactInput, prepared scanplan.PreparedAuditOperation) OpenAPIScanTask {
	input.Ref = copyArtifactRef(input.Ref)
	task := OpenAPIScanTask{Scanner: settings.Scanner(), Operation: prepared.Operation, Settings: input,
		PreparationDigest: prepared.PreparationDigest, Runnable: prepared.Runnable,
		TestParameters: settings.TestParameters(), Gaps: []string{}}
	if prepared.RequestSet != nil && len(prepared.RequestSet.Requests) == 1 {
		task.RequestDigest = prepared.RequestSet.Requests[0].ContentDigest
	}
	if prepared.Target != nil {
		task.TargetURL = prepared.Target.URL
	}
	seen := map[string]bool{}
	for _, gap := range prepared.Gaps {
		if !seen[gap.Code] {
			task.Gaps = append(task.Gaps, gap.Code)
			seen[gap.Code] = true
		}
	}
	sort.Strings(task.Gaps)
	return task
}

// PrepareOpenAPIScanTask validates retained bytes and reproduces the accepted
// task. References remain Project-scoped provenance; a Run fork can have a new
// ref but must contain the same bytes. Ownership/fork authorization is a caller
// responsibility, as with other preparation helpers.
func PrepareOpenAPIScanTask(task ItemTask, source, settingsData []byte) (scanplan.PreparedAuditOperation, error) {
	if validateItemTask(task) != nil || task.Scan == nil || DigestBytes(source) != task.SourceContentDigest ||
		DigestBytes(settingsData) != task.Scan.Settings.Digest {
		return scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "scan.inputs")
	}
	settings, err := scanplan.DecodeAuditScanSettings(settingsData)
	if err != nil {
		return scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "scan.settings")
	}
	prepared, err := settings.PrepareOperation(source, task.SourceMediaType, task.SourceRef, task.Scan.Operation)
	if err != nil || !sameCanonicalValue(task.Scan, openAPIScanTask(settings, task.Scan.Settings, prepared)) {
		return scanplan.PreparedAuditOperation{}, invalid(CodeInventoryInvalid, "scan.preparation")
	}
	return prepared, nil
}

func openAPIScanKey(inventoryDigest, pointer string) string {
	return "scan-" + strings.TrimPrefix(DigestBytes([]byte(inventoryDigest+"\x00"+pointer)), "sha256:")
}

func validateOpenAPIScanTask(value OpenAPIScanTask) error {
	if value.Scanner != "sqlmap" && value.Scanner != "nuclei" || !scanplan.ValidOperationPointer(value.Operation) ||
		validateIdentifier(value.Settings.Name, "scan.settings.name") != nil || value.Settings.Ref.ValidateExact() != nil ||
		!validDigest(value.Settings.Digest) || !validDigest(value.PreparationDigest) ||
		validateSortedStrings(value.Gaps, MaximumCoverageValues, "scan.gaps", false) != nil ||
		value.TestParameters == nil || !sort.StringsAreSorted(value.TestParameters) ||
		!value.Runnable && len(value.Gaps) == 0 {
		return invalid(CodeInvalid, "scan")
	}
	if contracts.ValidateScanTestParameters(value.TestParameters) != nil {
		return invalid(CodeInvalid, "scan.test_parameters")
	}
	if value.Scanner == "sqlmap" {
		if value.TargetURL != "" || len(value.TestParameters) == 0 ||
			(value.RequestDigest != "" || value.Runnable) && !validDigest(value.RequestDigest) {
			return invalid(CodeInvalid, "scan.request")
		}
	} else {
		request := contracts.PreparedHTTPRequest{Method: "GET", URL: value.TargetURL, Headers: []contracts.HTTPRequestHeader{}}
		if value.RequestDigest != "" || len(value.TestParameters) != 0 ||
			(value.TargetURL != "" || value.Runnable) && request.Validate() != nil ||
			!containsString(value.Gaps, "url_template_scan_only") {
			return invalid(CodeInvalid, "scan.target")
		}
	}
	return nil
}
