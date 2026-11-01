package artifacts

import (
	"context"
	"crypto/sha256"
	"encoding/hex"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// RetainAuditDependency preserves an exact baseline input in its Audit's
// protected Project namespace. The claim-bound Audit admission caller proves
// baseline membership before calling this purpose-specific primitive. Replay
// reconciles lineage, so a lost fork reply creates neither a new revision nor
// another evidence charge.
func (s *Service) RetainAuditDependency(ctx context.Context, projectID, namespace string, source contracts.ArtifactRef) (ForkResult, error) {
	scope, err := ProjectScope(projectID)
	if err != nil {
		return ForkResult{}, err
	}
	target := contracts.ArtifactRef{Namespace: namespace, Name: "dependency-" + exactAuditDependencyName(source)}
	if source.ValidateExact() != nil || !artifactpolicy.IsAuditManagedProjectNamespace(target.Namespace) || target.Validate() != nil {
		return ForkResult{}, ErrInvalidName
	}
	return s.Fork(ctx, Fork{Kind: LineageAuditImport, Source: scope, SourceRef: source, Target: scope, TargetRef: target, Frozen: true, Replay: true})
}

func exactAuditDependencyName(source contracts.ArtifactRef) string {
	digest := sha256.Sum256([]byte(source.Key()))
	return hex.EncodeToString(digest[:])
}
