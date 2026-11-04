package auditdomain

import (
	"errors"
	"slices"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

// ScanTask records an assigned scan, separately from operation tracing.
// Concrete request bytes and credentials stay in the exact supplied inputs.
// Re-preparation must match this identity before a scanner can be invoked.
type ScanTask struct {
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

func (s ScanTask) CoverageRequirement() string {
	if s.Scanner == "nuclei" {
		return "nuclei-url-template-scan"
	}
	return "sqlmap-request-scan"
}

func openAPIScanSubjects(source []byte, sourceMediaType string, settingsData []byte, settingsInput ExactInput, options InventoryOptions, maxItems int) (inventoryBasis, []inventorySubject, error) {
	if settingsInput.Name == options.SourceInputName ||
		validateIdentifier(settingsInput.Name, "settings.name") != nil || settingsInput.Ref.ValidateExact() != nil ||
		settingsInput.Digest != DigestBytes(settingsData) {
		return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "scan.settings")
	}
	settings, err := scanplan.DecodeAuditScanSettings(settingsData)
	if err != nil {
		return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "scan.settings")
	}
	basis := inventoryBasis{Schema: InventoryBasisSchema, Kind: "openapi-scans", Subjects: []map[string]any{}, Gaps: []string{}}
	subjects := []inventorySubject{}
	// One parse and one source digest serve every selected operation (SCN-R256).
	for prepared, err := range settings.PrepareOperations(source, normalizedMediaType(sourceMediaType), options.SourceRef) {
		if err != nil {
			var failure *scanplan.PreparationError
			if errors.As(err, &failure) {
				return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "scan."+failure.Code)
			}
			return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "scan.preparation")
		}
		if len(subjects) >= maxItems {
			return inventoryBasis{}, nil, invalid(CodeLimitExceeded, "scan.operations")
		}
		task := openAPIScanTask(settings, settingsInput, prepared)
		subjects = append(subjects, inventorySubject{kind: "openapi-scan", scan: &task,
			requested: []string{task.CoverageRequirement()}, gaps: task.Gaps})
		basis.Subjects = append(basis.Subjects, map[string]any{"scan": task})
	}
	canonical, err := canonicalJSON(basis)
	if err != nil {
		return inventoryBasis{}, nil, invalid(CodeInventoryInvalid, "scan.inventory")
	}
	canonicalDigest := DigestBytes(canonical)
	for index := range subjects {
		subjects[index].itemKey = openAPIScanKey(canonicalDigest, subjects[index].scan.Operation)
		subjects[index].subjectKey = subjects[index].itemKey
	}
	return basis, subjects, nil
}

func openAPIScanTask(settings scanplan.AuditScanSettings, input ExactInput, prepared scanplan.PreparedAuditOperation) ScanTask {
	input.Ref = input.Ref.Clone()
	task := ScanTask{Scanner: settings.Scanner(), Operation: prepared.Operation, Settings: input,
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

func openAPIScanKey(inventoryDigest, pointer string) string {
	return "scan-" + strings.TrimPrefix(DigestBytes([]byte(inventoryDigest+"\x00"+pointer)), "sha256:")
}

func validateScanTask(value ScanTask, web bool) error {
	operationValid := scanplan.ValidOperationPointer(value.Operation)
	if web {
		operationValid = value.Scanner == "nuclei" && value.Operation == "target" && value.Runnable
	}
	if value.Scanner != "sqlmap" && value.Scanner != "nuclei" || !operationValid ||
		validateIdentifier(value.Settings.Name, "scan.settings.name") != nil || value.Settings.Ref.ValidateExact() != nil ||
		!contentdigest.Valid(value.Settings.Digest) || !contentdigest.Valid(value.PreparationDigest) ||
		validateSortedStrings(value.Gaps, MaximumCoverageValues, "scan.gaps", false) != nil ||
		value.TestParameters == nil || !sort.StringsAreSorted(value.TestParameters) ||
		!value.Runnable && len(value.Gaps) == 0 {
		return invalid(CodeInvalid, "scan")
	}
	if scan.ValidateScanTestParameters(value.TestParameters) != nil {
		return invalid(CodeInvalid, "scan.test_parameters")
	}
	if value.Scanner == "sqlmap" {
		if value.TargetURL != "" || len(value.TestParameters) == 0 ||
			(value.RequestDigest != "" || value.Runnable) && !contentdigest.Valid(value.RequestDigest) {
			return invalid(CodeInvalid, "scan.request")
		}
	} else {
		request := scan.PreparedHTTPRequest{Method: "GET", URL: value.TargetURL, Headers: []scan.HTTPRequestHeader{}}
		if value.RequestDigest != "" || len(value.TestParameters) != 0 ||
			(value.TargetURL != "" || value.Runnable) && request.Validate() != nil ||
			!slices.Contains(value.Gaps, "url_template_scan_only") {
			return invalid(CodeInvalid, "scan.target")
		}
	}
	return nil
}
