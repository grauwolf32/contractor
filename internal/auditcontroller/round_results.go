package auditcontroller

import (
	"context"
	"errors"
	"strconv"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

var errRoundResultsPending = errors.New("settled Round snapshot construction made bounded progress")

// A bound child remains an ordinary submitted execution while its exact input
// closure is being built. Resume that same child on later Controller passes;
// waiting solely for terminal observation would strand it behind the fence.
func (c *Controller) resumeRoundResultsInput(ctx context.Context, claim auditstore.ControllerClaim, snapshot auditstore.ReconcileSnapshot, binding auditprofile.ResolvedAuditWorkflowBinding, execution auditstore.Execution) (bool, bool, error) {
	if execution.State != auditstore.ExecutionSubmitted || execution.RunID == nil || execution.RoleAttempt == nil || snapshot.RunStates[*execution.RunID] != runstore.RunInitializing {
		return false, false, nil
	}
	declared := false
	for _, mapping := range binding.Inputs {
		if mapping.Source == auditprofile.AuditInputFromRoundResults {
			declared = true
		}
	}
	if !declared {
		return false, false, nil
	}
	prepared, err := c.builder.WithClaim(claim).PrepareRole(ctx, snapshot, execution.WorkflowRole, *execution.RoleAttempt)
	if err != nil {
		return false, true, err
	}
	if prepared.Intent.ExecutionID != execution.ExecutionID || prepared.Intent.RequestDigest != execution.RequestDigest {
		return false, true, invalidSubmission("initializing assessment differs from its immutable execution")
	}
	changed, err := c.createRun(ctx, claim, prepared)
	return changed, true, err
}

// RoundResultsStore uses existing Audit controller ownership and retained
// artifact links. Each mutating call is one bounded capture/publication slice.
type RoundResultsStore interface {
	GetRoundResultsBuild(context.Context, auditstore.RoundResultsStepParams) (auditstore.RoundResultsBuild, error)
	AdvanceRoundResults(context.Context, auditstore.RoundResultsStepParams) (auditstore.RoundResultsBuild, error)
	ReadRoundResultsPage(context.Context, auditstore.RoundResultsStepParams, uint64, int) ([]byte, error)
	PublishRoundResultsPage(context.Context, auditstore.PublishRoundResultsPageParams) error
	PublishRoundResultsManifest(context.Context, auditstore.PublishRoundResultsManifestParams) error
}

func (b *PinnedSubmissionBuilder) WithRoundResultsStore(store RoundResultsStore) *PinnedSubmissionBuilder {
	bound := *b
	bound.roundResults = store
	return &bound
}

func (b *PinnedSubmissionBuilder) prepareRoundResults(ctx context.Context, snapshot auditstore.ReconcileSnapshot) (auditstore.ExactArtifact, error) {
	if b.roundResults == nil || snapshot.Round == nil || snapshot.Round.State != auditstore.RoundAssessing {
		return auditstore.ExactArtifact{}, invalidSubmission("round-results requires the settled assessment barrier")
	}
	p := auditstore.RoundResultsStepParams{Claim: b.claim, RoundID: snapshot.Round.RoundID}
	state, err := b.roundResults.GetRoundResultsBuild(ctx, p)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if state.Manifest != nil {
		return state.Manifest.Clone(), nil
	}
	if !state.Sealed {
		_, err = b.roundResults.AdvanceRoundResults(ctx, p)
		return auditstore.ExactArtifact{}, roundResultsProgress(err)
	}
	manifest := auditdomain.RoundResultsManifest{Schema: auditdomain.RoundResultsSchema, AuditID: state.AuditID,
		RoundID: state.RoundID, SourceGeneration: state.Generation, ItemCount: state.ItemCount,
		MetadataBytes: state.MetadataBytes, Pages: make([]auditdomain.RoundResultsPageRef, 0, len(state.Pages))}
	for _, page := range state.Pages {
		if page.Artifact == nil {
			content, err := b.roundResults.ReadRoundResultsPage(ctx, p, state.Generation, page.Ordinal)
			if err != nil {
				return auditstore.ExactArtifact{}, err
			}
			decoded, err := auditdomain.DecodeRoundResultsPage(content)
			if err != nil || decoded.AuditID != state.AuditID || decoded.RoundID != state.RoundID || decoded.SourceGeneration != state.Generation || decoded.Ordinal != page.Ordinal {
				return auditstore.ExactArtifact{}, invalidSubmission("captured Round results page is invalid")
			}
			artifact, err := b.artifacts.PutImmutableProject(ctx, snapshot.Audit.ProjectID,
				contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(state.AuditID), Name: "round-results-" + stringsDigest(auditdomain.DigestBytes(content))},
				artifacts.Payload{MediaType: "application/json", Data: content})
			if err != nil {
				return auditstore.ExactArtifact{}, err
			}
			err = b.roundResults.PublishRoundResultsPage(ctx, auditstore.PublishRoundResultsPageParams{RoundResultsStepParams: p, Generation: state.Generation, Ordinal: page.Ordinal, Artifact: artifact})
			return auditstore.ExactArtifact{}, roundResultsProgress(err)
		}
		manifest.Pages = append(manifest.Pages, auditdomain.RoundResultsPageRef{Ordinal: page.Ordinal, ItemCount: page.ItemCount, Artifact: page.Artifact.Clone()})
	}
	content, err := auditdomain.EncodeRoundResultsManifest(manifest)
	if err != nil {
		return auditstore.ExactArtifact{}, invalidSubmission("complete Round results manifest exceeds its contract")
	}
	artifact, err := b.artifacts.PutImmutableProject(ctx, snapshot.Audit.ProjectID,
		contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(state.AuditID), Name: "round-results-root-" + strconv.FormatUint(state.Generation, 10) + "-" + stringsDigest(auditdomain.DigestBytes(content))}, artifacts.Payload{MediaType: "application/json", Data: content})
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	err = b.roundResults.PublishRoundResultsManifest(ctx, auditstore.PublishRoundResultsManifestParams{RoundResultsStepParams: p, Generation: state.Generation, Manifest: artifact, Content: manifest})
	return auditstore.ExactArtifact{}, roundResultsProgress(err)
}

func roundResultsProgress(err error) error {
	if err == nil {
		return errRoundResultsPending
	}
	if errors.Is(err, auditstore.ErrInvalid) || errors.Is(err, auditstore.ErrEvidenceBudgetExhausted) {
		return invalidSubmission("complete Round results snapshot cannot be admitted")
	}
	return err
}
