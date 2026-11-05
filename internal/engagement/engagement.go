// Package engagement prepares the exact public policy Artifact for controlled
// allocation execution. It owns no durable policy store or credential lifecycle.
package engagement

import (
	"context"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// Pin is a transient, authorized read of an existing exact Artifact. It is not
// another persistent snapshot; its producer supplies trusted authorization facts.
type Pin struct {
	Artifact     contracts.ExactArtifact
	Document     engagementcontract.Document
	AuthorizedBy string
	AuthorizedAt time.Time
}

// InputSlot recognizes the explicit type declaration, independently of its name.
// A mixed-media or second policy slot makes execution authority ambiguous.
func InputSlot(inputs map[string]resolved.ArtifactSlot) (string, error) {
	selected := ""
	for name, slot := range inputs {
		for _, media := range slot.MediaTypes {
			if media != engagementcontract.MediaType {
				continue
			}
			if len(slot.MediaTypes) != 1 || selected != "" {
				return "", contracts.Invalidf("engagement needs one explicit typed input slot")
			}
			selected = name
		}
	}
	return selected, nil
}

// Read checks metadata before materializing the bounded exact policy bytes.
func Read(ctx context.Context, store artifacts.ScopedStore, exact contracts.ExactArtifact) (engagementcontract.Document, error) {
	if exact.Validate() != nil || !exact.HasMetadata() || exact.MediaType != engagementcontract.MediaType || exact.SizeBytes > engagementcontract.MaxDocumentBytes {
		return engagementcontract.Document{}, contracts.Invalidf("engagement input metadata is invalid")
	}
	read, err := store.ReadExact(ctx, exact)
	if err != nil {
		return engagementcontract.Document{}, err
	}
	document, err := engagementcontract.Decode(read.Payload.Data)
	if err != nil {
		return engagementcontract.Document{}, contracts.Invalidf("engagement failed validation")
	}
	return document, nil
}

// CredentialIDs is the bounded non-secret dependency set of a validated policy.
func CredentialIDs(document engagementcontract.Document) []string {
	ids := []string{}
	for _, identity := range document.Identities {
		if identity.Credential != nil {
			ids = append(ids, identity.Credential.CredentialID)
		}
	}
	sort.Strings(ids)
	return ids
}

// Project assembles the same narrow assignment for every authorized producer.
// Trusted infrastructure resolution must have finished outside database locks.
func Project(pin Pin, usesHTTP bool, protected []engagementcontract.ProtectedDestination) (*engagementcontract.AllocationProjection, error) {
	result := &engagementcontract.AllocationProjection{
		Schema:     "contractor.engagement.allocation.v1",
		Engagement: pin.Artifact.Ref, EngagementDigest: pin.Artifact.Digest,
		AuthorizedBy: pin.AuthorizedBy, AuthorizedAt: pin.AuthorizedAt,
	}
	if usesHTTP {
		result.HTTP = &engagementcontract.HTTPAssignment{
			Adapter: runtimesettings.RuntimeAdapterHTTPTarget,
			Routes:  pin.Document.Egress.Routes, Exclusions: pin.Document.Egress.Exclusions,
			ProtectedDestinations: protected, Identities: pin.Document.Identities,
			Limits: pin.Document.Limits,
		}
	}
	if err := result.Validate(); err != nil {
		return nil, err
	}
	return result, nil
}
