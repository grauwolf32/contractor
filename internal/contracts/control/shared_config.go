package control

import (
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/pagelimit"
)

// SharedConfig is the immutable startup policy fetched before registration.
// Only cross-component policy belongs here; allocation settings remain separate.
type SharedConfig struct {
	APIVersion string       `json:"apiVersion" strictjson:"required"`
	Version    int          `json:"version" strictjson:"required"`
	Limits     SharedLimits `json:"limits" strictjson:"required"`
	Digest     string       `json:"digest" strictjson:"required"`
}

type SharedLimits struct {
	GitImportUrlMaxBytes     int `json:"gitImportUrlMaxBytes" strictjson:"required"`
	RunMetadataMaxLabels     int `json:"runMetadataMaxLabels" strictjson:"required"`
	RunMetadataKeyMaxBytes   int `json:"runMetadataKeyMaxBytes" strictjson:"required"`
	RunMetadataValueMaxBytes int `json:"runMetadataValueMaxBytes" strictjson:"required"`
	AuditMaxLabels           int `json:"auditMaxLabels" strictjson:"required"`

	HttpRequestBodyMaxBytes  int `json:"httpRequestBodyMaxBytes" strictjson:"required"`
	HttpResponseBodyMaxBytes int `json:"httpResponseBodyMaxBytes" strictjson:"required"`
	HttpHeaderBlockMaxBytes  int `json:"httpHeaderBlockMaxBytes" strictjson:"required"`

	WorkerResultMaxBytes              int `json:"workerResultMaxBytes" strictjson:"required"`
	WorkerCompletionMaxBytes          int `json:"workerCompletionMaxBytes" strictjson:"required"`
	WorkerFailureMessageMaxBytes      int `json:"workerFailureMessageMaxBytes" strictjson:"required"`
	WorkerResultMaxArtifacts          int `json:"workerResultMaxArtifacts" strictjson:"required"`
	WorkerObservationMaxTools         int `json:"workerObservationMaxTools" strictjson:"required"`
	WorkerWorkspaceMaxFilesRead       int `json:"workerWorkspaceMaxFilesRead" strictjson:"required"`
	WorkerStateMaxBytes               int `json:"workerStateMaxBytes" strictjson:"required"`
	WorkerStateWorkspaceMaxPaths      int `json:"workerStateWorkspaceMaxPaths" strictjson:"required"`
	WorkerStateWorkspacePathsMaxBytes int `json:"workerStateWorkspacePathsMaxBytes" strictjson:"required"`

	ArtifactPayloadMaxBytes int64 `json:"artifactPayloadMaxBytes" strictjson:"required"`
	PublicPageMaxItems      int   `json:"publicPageMaxItems" strictjson:"required"`
	ArtifactListMaxItems    int   `json:"artifactListMaxItems" strictjson:"required"`
}

// DefaultSharedConfig derives policy from the Server-owned definitions.
func DefaultSharedConfig() (SharedConfig, error) {
	c := SharedConfig{APIVersion: contracts.APIVersion, Version: 1,
		Limits: SharedLimits{
			GitImportUrlMaxBytes:     contracts.MaxGitImportURLBytes,
			RunMetadataMaxLabels:     runlabels.MaxRunMetadataLabels,
			RunMetadataKeyMaxBytes:   runlabels.MaxRunMetadataLabelKey,
			RunMetadataValueMaxBytes: runlabels.MaxRunMetadataLabelValue,
			AuditMaxLabels:           runlabels.MaxAuditLabels,

			HttpRequestBodyMaxBytes:  contracts.MaxHTTPRequestBodyBytes,
			HttpResponseBodyMaxBytes: contracts.MaxHTTPResponseBodyBytes,
			HttpHeaderBlockMaxBytes:  contracts.MaxHTTPHeaderBlockBytes,

			WorkerResultMaxBytes:              contracts.MaxWorkerResultBytes,
			WorkerCompletionMaxBytes:          contracts.MaxWorkerCompletionBytes,
			WorkerFailureMessageMaxBytes:      contracts.MaxWorkerFailureMessageBytes,
			WorkerResultMaxArtifacts:          contracts.MaxWorkerResultArtifacts,
			WorkerObservationMaxTools:         contracts.MaxWorkerObservationTools,
			WorkerWorkspaceMaxFilesRead:       contracts.MaxWorkerFilesRead,
			WorkerStateMaxBytes:               reporting.MaxAgentStateSnapshotBytes,
			WorkerStateWorkspaceMaxPaths:      reporting.MaxStateWorkspacePaths,
			WorkerStateWorkspacePathsMaxBytes: reporting.MaxStateWorkspacePathBytes,

			ArtifactPayloadMaxBytes: contracts.MaxArtifactPayloadBytes,
			PublicPageMaxItems:      pagelimit.MaxRows,
			ArtifactListMaxItems:    contracts.MaxArtifactListItems,
		}}
	var err error
	c.Digest, err = c.digest()
	if err != nil {
		return SharedConfig{}, err
	}
	return c, c.Validate()
}

func (c SharedConfig) digest() (string, error) {
	// Hash every field of the typed limits projection in canonical key order.
	return contentdigest.CanonicalJSON(struct {
		Limits  SharedLimits `json:"limits"`
		Version int          `json:"version"`
	}{c.Limits, c.Version})
}

func (c SharedConfig) Validate() error {
	if err := contracts.ValidateAPIVersion(c.APIVersion); err != nil {
		return err
	}
	if c.Version != 1 {
		return contracts.Invalidf("unsupported shared config version")
	}
	for _, limit := range []int64{int64(c.Limits.GitImportUrlMaxBytes), int64(c.Limits.RunMetadataMaxLabels), int64(c.Limits.RunMetadataKeyMaxBytes), int64(c.Limits.RunMetadataValueMaxBytes), int64(c.Limits.AuditMaxLabels), int64(c.Limits.HttpRequestBodyMaxBytes), int64(c.Limits.HttpResponseBodyMaxBytes), int64(c.Limits.HttpHeaderBlockMaxBytes), int64(c.Limits.WorkerResultMaxBytes), int64(c.Limits.WorkerCompletionMaxBytes), int64(c.Limits.WorkerFailureMessageMaxBytes), int64(c.Limits.WorkerResultMaxArtifacts), int64(c.Limits.WorkerObservationMaxTools), int64(c.Limits.WorkerWorkspaceMaxFilesRead), int64(c.Limits.WorkerStateMaxBytes), int64(c.Limits.WorkerStateWorkspaceMaxPaths), int64(c.Limits.WorkerStateWorkspacePathsMaxBytes), c.Limits.ArtifactPayloadMaxBytes, int64(c.Limits.PublicPageMaxItems), int64(c.Limits.ArtifactListMaxItems)} {
		if limit <= 0 || limit > contracts.MaxExactInteger {
			return contracts.Invalidf("invalid shared config limit")
		}
	}
	digest, err := c.digest()
	if err != nil {
		return err
	}
	if c.Digest != digest {
		return contracts.Invalidf("invalid shared config digest")
	}
	return nil
}
