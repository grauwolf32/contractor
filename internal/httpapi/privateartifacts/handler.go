// Package privateartifacts exposes allocation-bound RunScope access and
// explicitly granted Project Artifact reads to mTLS-authenticated Runtime Agents.
package privateartifacts

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
	"github.com/grauwolf32/contractor/internal/httpapi/artifacttransfer"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const RuntimeInstanceHeader = "X-Contractor-Runtime-Instance-ID"

type AllocationRegistry interface {
	GetGrant(string) (controlplane.AllocationGrant, error)
	WithWriteGrant(string, func(controlplane.AllocationGrant) error) error
}

type Dependencies struct {
	PlannerOperations PlannerOperations
	Transfers         artifacttransfer.Admission
	GatewayRecovery   *gatewayrecovery.Service
	Registry          AllocationRegistry
	Artifacts         *artifacts.Service
	Findings          FindingIntake
	AuditEvidence     AuditEvidenceIntake
	AuditJudgments    AuditJudgmentIntake
	ProjectFindings   *findingintake.CollectionPublisher
	NewRequestID      func() (string, error)
	Logger            *slog.Logger
	Sitemap           *projectsitemap.Service
	RunProjects       RunProjectReader
	Projects          ProjectReader
}

type RunProjectReader interface {
	ProjectContext(context.Context, string) (runstore.ProjectContext, error)
}

type ProjectReader interface {
	Get(context.Context, string, string) (projectstore.Project, error)
}

type AuditEvidenceIntake interface {
	RecordEvidence(context.Context, controlplane.AllocationGrant, auditcontract.EvidenceSubmission) (auditcontract.EvidenceReceipt, error)
}

type AuditJudgmentIntake interface {
	RecordJudgments(context.Context, controlplane.AllocationGrant, auditcontract.JudgmentSubmission) (auditcontract.JudgmentReceipt, error)
}

type FindingIntake interface {
	FindReplay(context.Context, controlplane.AllocationGrant, findingintake.Submission) (findingintake.Receipt, bool, error)
	Submit(context.Context, controlplane.AllocationGrant, findingintake.Submission) (findingintake.Receipt, bool, error)
}

type handler struct {
	dependencies Dependencies
	recovery     gatewayRecoveryUpdater
}

type authenticatedRuntime struct {
	principalID string
	instanceID  string
}

func (h *handler) runStoreForGrant(
	allocationID string,
	grant controlplane.AllocationGrant,
	identity authenticatedRuntime,
	write bool,
) (artifacts.ScopedStore, error) {
	if grant.AllocationID != allocationID || strings.TrimSpace(grant.RunID) == "" ||
		grant.RuntimeAgentID != identity.principalID || grant.RuntimeInstanceID != identity.instanceID {
		return artifacts.ScopedStore{}, controlplane.ErrAllocationNotFound
	}
	if grant.ReadPolicy != controlplane.ReadCurrentRun {
		return artifacts.ScopedStore{}, errArtifactAccessDenied
	}
	if write {
		if grant.WriteFenced {
			return artifacts.ScopedStore{}, errAllocationWriteFenced
		}
		if grant.WritePolicy != controlplane.WriteInputsAndIntermediates {
			return artifacts.ScopedStore{}, errArtifactAccessDenied
		}
	}
	store, err := h.dependencies.Artifacts.Run(grant.RunID)
	if err != nil {
		return artifacts.ScopedStore{}, fmt.Errorf("bind allocation RunScope: %w", err)
	}
	return store, nil
}
