package auditservice

import (
	"context"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// pinAssertionSource interprets one finite selected domain input at Start.
// Its frozen plan is bounded by 128 entries/1 MiB; admission still commits
// smaller common mutations and never invokes a model to generate seeds.
func pinAssertionSource(ctx context.Context, art *artifacts.Service, audit auditstore.Audit, profile auditprofile.PinnedProgram, baseline *BaselineSnapshot) error {
	selection := profile.Program.SeedSource
	if selection == nil {
		return nil
	}
	var plan auditcontract.InitialAssertions
	var err error
	var sourceScope artifacts.Scope
	var exact contracts.ExactArtifact
	if selection.Standard != nil {
		catalog, err := auditstandards.NewCatalog(art)
		if err != nil {
			return err
		}
		pkg, err := catalog.Resolve(ctx, audit.OwnerID, auditstandards.Reference{Scheme: selection.Standard.Scheme, Version: selection.Standard.Version})
		if err != nil {
			return err
		}
		plan, err = auditdomain.StandardAssertions(pkg.Package, *selection)
		if err != nil {
			return err
		}
		sourceScope, err = artifacts.UserScope(audit.OwnerID)
		if err != nil {
			return err
		}
		exact = contracts.ExactArtifact{Ref: pkg.Source.Artifact, Digest: pkg.Source.Digest, MediaType: pkg.Source.MediaType, SizeBytes: pkg.Source.SizeBytes}
		baseline.Standards = append(baseline.Standards, auditstandards.PinnedPackage{Reference: pkg.Package.Reference(), Title: pkg.Package.Document.Standard.Title, Source: pkg.Package.Document.Standard.Source, License: pkg.Package.Document.Standard.License, Catalog: pkg.Source})
	} else {
		var found bool
		exact, found = baseline.Inputs[selection.Input]
		if !found || exact.SizeBytes > auditcontract.MaxMutationBytes {
			return ErrInvalid
		}
		project, err := art.Project(audit.ProjectID)
		if err != nil {
			return err
		}
		read, err := project.Read(ctx, exact.Ref)
		if err != nil {
			return err
		}
		if exact.Digest != auditdomain.DigestBytes(read.Payload.Data) || exact.SizeBytes != int64(len(read.Payload.Data)) {
			return ErrInvalid
		}
		plan, err = auditdomain.ChecklistAssertions(read.Payload.Data, exact.MediaType, selection.Workflow)
		if err != nil {
			return err
		}
		sourceScope, err = artifacts.ProjectScope(audit.ProjectID)
		if err != nil {
			return err
		}
	}
	target, _ := artifacts.AuditScope(audit.AuditID)
	retained, err := art.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageAuditInput, Source: sourceScope, SourceRef: exact.Ref, Target: target, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: "assertion-source"}, Frozen: true})
	if err != nil {
		return err
	}
	plan.Source = auditcontract.Material{ScopeKind: "audit", ScopeID: audit.AuditID, Ref: retained.TargetRef, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
	if len(baseline.Standards) > 0 {
		baseline.Standards[len(baseline.Standards)-1].Retained = auditstandards.ExactPackage{Artifact: retained.TargetRef, Digest: exact.Digest, MediaType: exact.MediaType, SizeBytes: exact.SizeBytes}
	}
	baseline.InitialAssertions = &plan
	return nil
}

func executionProfile(audit auditstore.Audit) (auditprofile.PinnedProgram, error) {
	pinned, err := auditprofile.DecodePinnedProgram(audit.ProfileSnapshot)
	if err != nil {
		return pinned, err
	}
	pinned.Program, err = auditdomain.ExecutionProgram(pinned.Program, audit.BaselineSnapshot)
	return pinned, err
}
