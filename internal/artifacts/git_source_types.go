package artifacts

import "time"

type GitSource struct {
	Analysis       *GitAnalysis `json:"analysis,omitempty"`
	RepositoryURL  string       `json:"repositoryUrl"`
	RequestedRef   *string      `json:"requestedRef"`
	ResolvedCommit string       `json:"resolvedCommit"`
	ImportedAt     time.Time    `json:"importedAt"`
}
