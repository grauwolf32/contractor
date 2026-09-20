package contracts

import "regexp"

const (
	MaxScanTestParameters         = 64
	MaxScanTestParameterNameBytes = 128
)

// ScanPlanPolicy bounds selected prepared inputs, jobs and the sum of fixed
// Worker timeouts. These are not limits on physical scanner HTTP requests.
type ScanPlanPolicy struct {
	InputArtifact   string           `json:"inputArtifact" yaml:"inputArtifact"`
	MaxInputs       int              `json:"maxInputs" yaml:"maxInputs"`
	MaxJobs         int              `json:"maxJobs" yaml:"maxJobs"`
	MaxTotalSeconds int              `json:"maxTotalSeconds" yaml:"maxTotalSeconds"`
	Tools           []ScanToolPolicy `json:"tools" yaml:"tools"`
}

type ScanToolPolicy struct {
	Worker           string   `json:"worker" yaml:"worker"`
	MaxJobs          int      `json:"maxJobs" yaml:"maxJobs"`
	MaxTotalSeconds  int      `json:"maxTotalSeconds" yaml:"maxTotalSeconds"`
	TestParameters   []string `json:"testParameters,omitempty" yaml:"testParameters,omitempty"`
	WordlistArtifact string   `json:"wordlistArtifact,omitempty" yaml:"wordlistArtifact,omitempty"`
}

var scanTestParameterPattern = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.\[\]-]*$`)

func (p ScanPlanPolicy) Validate() error {
	if ValidateArtifactName(p.InputArtifact) != nil || p.MaxInputs < 1 || p.MaxInputs > 1000 || p.MaxJobs < 1 || p.MaxJobs > 100 || p.MaxTotalSeconds < 1 || p.MaxTotalSeconds > 86400 || len(p.Tools) < 1 || len(p.Tools) > 4 {
		return invalidf("invalid scan plan policy or budget")
	}
	workers := map[string]bool{}
	for _, tool := range p.Tools {
		if ValidateArtifactName(tool.Worker) != nil || workers[tool.Worker] || tool.MaxJobs < 1 || tool.MaxJobs > 100 || tool.MaxTotalSeconds < 1 || tool.MaxTotalSeconds > 86400 {
			return invalidf("invalid scan tool policy or duplicate worker")
		}
		workers[tool.Worker] = true
		if tool.WordlistArtifact != "" && ValidateArtifactName(tool.WordlistArtifact) != nil {
			return invalidf("invalid scan wordlist artifact slot")
		}
		if err := ValidateScanTestParameters(tool.TestParameters); err != nil {
			return err
		}
	}
	return nil
}

// ValidateScanTestParameters is shared by authored policies and retained Audit
// settings. An empty list is valid here; SQLMap's caller requires a selection.
func ValidateScanTestParameters(names []string) error {
	if len(names) > MaxScanTestParameters {
		return invalidf("too many scan test parameters")
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if len(name) > MaxScanTestParameterNameBytes || !scanTestParameterPattern.MatchString(name) || seen[name] {
			return invalidf("invalid or duplicate scan test parameter")
		}
		seen[name] = true
	}
	return nil
}
