package gitimport

import (
	"encoding/json"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

type Mode string

const (
	ModeSnapshot           Mode = "snapshot"
	ModeAnalysis           Mode = "analysis"
	maxAnalysisReportBytes      = 4 << 20
)

type exclusion struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

type analysisReport struct {
	Policy         string      `json:"policy"`
	ResolvedCommit string      `json:"resolvedCommit"`
	IncludedFiles  int         `json:"includedFiles"`
	Excluded       []exclusion `json:"excluded"`
	encodedBytes   int
}

func (r *analysisReport) exclude(path, reason string) error {
	entry := exclusion{Path: path, Reason: reason}
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	// Bound the list before allocating the final report. encode checks the
	// complete JSON, including its small fixed header, against the same ceiling.
	if len(data)+1 > maxAnalysisReportBytes-r.encodedBytes {
		return ErrBudget
	}
	r.encodedBytes += len(data) + 1
	r.Excluded = append(r.Excluded, entry)
	return nil
}

func (r *analysisReport) encode(commit string, files int) ([]byte, *artifacts.GitAnalysis, error) {
	r.Policy = artifacts.GitAnalysisPolicy
	r.ResolvedCommit = commit
	r.IncludedFiles = files
	sort.Slice(r.Excluded, func(i, j int) bool { return r.Excluded[i].Path < r.Excluded[j].Path })
	data, err := json.Marshal(r)
	if err != nil {
		return nil, nil, err
	}
	if len(data) > maxAnalysisReportBytes {
		return nil, nil, ErrBudget
	}
	return data, &artifacts.GitAnalysis{Policy: r.Policy, ReportPath: artifacts.GitAnalysisReportPath, IncludedFiles: files, ExcludedEntries: len(r.Excluded)}, nil
}
