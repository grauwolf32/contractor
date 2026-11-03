package projectstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// EditStore keeps patch application on the same ordinary Project CAS writer.
type EditStore interface {
	Get(context.Context, string, string) (Project, error)
	Update(context.Context, UpdateParams) (Project, error)
}

// EditCredentialGuard is the existing credential lifecycle barrier; a target
// cannot attach a credential while another operation deletes it.
type EditCredentialGuard interface {
	WithCredentialReferences(context.Context, func() error) error
}

type EditParams struct {
	OwnerID             string
	OperationsPrincipal bool
	ProjectID           string
	ExpectedRevision    uint64
	Name, Description   *string
	HTTPTarget          *runtimesettings.HTTPOriginTargetRef
	SetHTTPTarget       bool
	ActiveChecks        *contracts.ActiveChecksPolicy
}

// CredentialValidation adapts the credential owner without importing its storage.
type CredentialValidation func(ctx context.Context, ownerID string, operations bool, credentialID, kind string) error

type Editor struct {
	projects    EditStore
	credentials EditCredentialGuard
	validate    CredentialValidation
}

func NewEditor(projects EditStore, credentialGuard EditCredentialGuard, validate CredentialValidation) *Editor {
	return &Editor{projects: projects, credentials: credentialGuard, validate: validate}
}

// Edit preserves omitted fields, explicitly clears a selected nil target and
// validates credential use under the shared barrier before the revision CAS.
func (e *Editor) Edit(ctx context.Context, p EditParams) (Project, error) {
	current, err := e.projects.Get(ctx, p.OwnerID, p.ProjectID)
	if err != nil {
		return Project{}, err
	}
	name, description := current.Name, current.Description
	if p.Name != nil {
		name = *p.Name
	}
	if p.Description != nil {
		description = *p.Description
	}
	target := current.HTTPTarget.Clone()
	if p.SetHTTPTarget {
		target = p.HTTPTarget.Clone()
	}
	var result Project
	update := func() error {
		var err error
		result, err = e.projects.Update(ctx, UpdateParams{
			ProjectID: p.ProjectID, OwnerID: p.OwnerID, ExpectedRevision: p.ExpectedRevision,
			Name: name, Description: description, HTTPTarget: target, ActiveChecks: p.ActiveChecks,
		})
		return err
	}
	if p.SetHTTPTarget && target != nil && target.Credential != nil {
		err = e.credentials.WithCredentialReferences(ctx, func() error {
			if err := e.validate(ctx, p.OwnerID, p.OperationsPrincipal, target.Credential.CredentialID, string(target.Credential.Kind)); err != nil {
				return err
			}
			return update()
		})
	} else {
		err = update()
	}
	return result, err
}
