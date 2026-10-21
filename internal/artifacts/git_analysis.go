package artifacts

import "github.com/grauwolf32/contractor/internal/strictjson"

// GitAnalysis is the immutable coverage summary of an explicitly filtered import.
// The ZIP member contains the complete per-path report, including its policy.
type GitAnalysis struct {
	Policy          string `json:"policy" strictjson:"required"`
	ReportPath      string `json:"reportPath" strictjson:"required"`
	IncludedFiles   int    `json:"includedFiles" strictjson:"required"`
	ExcludedEntries int    `json:"excludedEntries" strictjson:"required"`
}

const GitAnalysisPolicy = "source-analysis-v1"
const GitAnalysisReportPath = ".contractor-import-report.json"

func (a GitAnalysis) valid() bool {
	return a.Policy == GitAnalysisPolicy && a.ReportPath == GitAnalysisReportPath && a.IncludedFiles >= 1 && a.IncludedFiles < 50000 && a.ExcludedEntries >= 0 && a.ExcludedEntries <= 50000
}

// UnmarshalJSON validates the persisted and public shape on every metadata read.
func (a *GitAnalysis) UnmarshalJSON(data []byte) error {
	type plain GitAnalysis
	var value plain
	if err := strictjson.Decode(data, &value); err != nil {
		return err
	}
	if !GitAnalysis(value).valid() {
		return ErrArtifactIntegrity
	}
	*a = GitAnalysis(value)
	return nil
}
