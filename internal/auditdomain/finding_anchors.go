package auditdomain

import (
	"bytes"
	"encoding/json"
	"path"
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	FindingAnchorDerivationSchema       = "contractor.finding.anchors-derived.v1"
	FindingAnchorProjectionSchema       = "contractor.finding.anchors-projection.v1"
	FindingAnchorGrammar                = "tree-sitter-language-pack/1.14.3;tree-sitter/0.25.2"
	MaximumFindingAnchorDerivationBytes = 16 << 10
	MaximumFindingAnchorProjectionBytes = 32 << 10
	MaximumFindingAnchors               = 32
	findingAnchorLocalRule              = "source.private-key-literal.ed25519/v1"
)

// FindingAnchorDerivation is carried only by the authenticated Runtime
// publisher, never by a model tool argument or a generic artifact write.
type FindingAnchorDerivation struct {
	Schema       string                    `json:"schema" strictjson:"required"`
	Availability string                    `json:"availability" strictjson:"required"`
	Reason       string                    `json:"reason,omitempty"`
	SourceRef    *contracts.ArtifactRef    `json:"sourceRef,omitempty"`
	SourceDigest string                    `json:"sourceDigest,omitempty"`
	Anchors      []FindingStructuralAnchor `json:"anchors,omitempty"`
}

type FindingStructuralAnchor struct {
	File            string                `json:"file" strictjson:"required"`
	StartLine       int64                 `json:"startLine" strictjson:"required"`
	EndLine         int64                 `json:"endLine" strictjson:"required"`
	Language        string                `json:"language" strictjson:"required"`
	Profile         string                `json:"profile" strictjson:"required"`
	Grammar         string                `json:"grammar" strictjson:"required"`
	FileDigest      string                `json:"fileDigest" strictjson:"required"`
	SignatureDigest string                `json:"signatureDigest" strictjson:"required"`
	Path            []int                 `json:"path" strictjson:"required"`
	NodeDigest      string                `json:"nodeDigest" strictjson:"required"`
	Context         *FindingAnchorContext `json:"context,omitempty"`
}

type FindingAnchorContext struct {
	Scheme        string `json:"scheme" strictjson:"required"`
	RuleID        string `json:"ruleId" strictjson:"required"`
	LiteralDigest string `json:"literalDigest" strictjson:"required"`
	ContextDigest string `json:"contextDigest" strictjson:"required"`
}

type FindingAnchorKeys struct {
	SiteKey     string `json:"siteKey" strictjson:"required"`
	IssueKey    string `json:"issueKey,omitempty"`
	IssueReason string `json:"issueReason,omitempty"`
}

// FindingAnchorProjection retains the authenticated basis and the Server's
// canonical keys on the existing receipt. It never sets human triage state.
type FindingAnchorProjection struct {
	Schema       string                   `json:"schema" strictjson:"required"`
	Availability string                   `json:"availability" strictjson:"required"`
	Reason       string                   `json:"reason,omitempty"`
	Repository   string                   `json:"repository,omitempty"`
	Commit       string                   `json:"commit,omitempty"`
	Source       *contracts.ExactArtifact `json:"source,omitempty"`
	Basis        *FindingAnchorDerivation `json:"basis,omitempty"`
	Keys         []FindingAnchorKeys      `json:"keys,omitempty"`
}

func UnavailableFindingAnchors(reason string) FindingAnchorProjection {
	return FindingAnchorProjection{Schema: FindingAnchorProjectionSchema, Availability: "unavailable", Reason: reason}
}

func DecodeFindingAnchorDerivation(data []byte) (FindingAnchorDerivation, error) {
	var value FindingAnchorDerivation
	if len(data) > MaximumFindingAnchorDerivationBytes {
		return value, invalid(CodeLimitExceeded, "derived_anchors")
	}
	if err := strictjson.DecodeNonNull(data, &value); err != nil {
		return value, err
	}
	if err := validateAnchorDerivation(value); err != nil {
		return value, err
	}
	var fields map[string]json.RawMessage
	if err := strictjson.Decode(data, &fields); err != nil {
		return value, err
	}
	if value.Availability == "unavailable" && len(fields) != 3 || value.Availability == "available" && len(fields) != 5 {
		return value, invalid(CodeInvalid, "derived_anchors")
	}
	if err := exactAnchorShape(data, value); err != nil {
		return value, err
	}
	return value, nil
}

func validateAnchorDerivation(value FindingAnchorDerivation) error {
	if value.Schema != FindingAnchorDerivationSchema {
		return invalid(CodeInvalid, "derived_anchors.schema")
	}
	if value.Availability == "unavailable" {
		if value.Reason == "not_recorded" || value.Reason == "discarded" || !validAnchorReason(value.Reason) || value.SourceRef != nil || value.SourceDigest != "" || len(value.Anchors) != 0 {
			return invalid(CodeInvalid, "derived_anchors")
		}
		return nil
	}
	if value.Availability != "available" || value.Reason != "" || value.SourceRef == nil || value.SourceRef.ValidateExact() != nil || !contentdigest.Valid(value.SourceDigest) || len(value.Anchors) == 0 || len(value.Anchors) > MaximumFindingAnchors {
		return invalid(CodeInvalid, "derived_anchors")
	}
	for _, anchor := range value.Anchors {
		if err := validateFindingSourceLocation(FindingLocation{File: anchor.File, Range: &FindingLineRange{StartLine: anchor.StartLine, EndLine: anchor.EndLine}}); err != nil {
			return err
		}
		language := map[string]string{".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript"}[path.Ext(anchor.File)]
		if len(anchor.File) > 512 || len(strings.Split(anchor.File, "/")) > 32 || language == "" || anchor.Language != language || anchor.Profile != language+"/statement-v1" || anchor.Grammar != FindingAnchorGrammar || !contentdigest.Valid(anchor.FileDigest) || !contentdigest.Valid(anchor.NodeDigest) || !contentdigest.Valid(anchor.SignatureDigest) || len(anchor.Path) == 0 || len(anchor.Path) > 128 {
			return invalid(CodeInvalid, "derived_anchors.anchor")
		}
		for _, index := range anchor.Path {
			if index < 0 || index > 100000 {
				return invalid(CodeInvalid, "derived_anchors.path")
			}
		}
		if c := anchor.Context; c != nil {
			want, err := contentdigest.CanonicalJSON([]any{"node-local-private-key/v1", "ed25519", c.LiteralDigest})
			if err != nil || c.Scheme != "node-local/v1" || c.RuleID != findingAnchorLocalRule || !contentdigest.Valid(c.LiteralDigest) || c.ContextDigest != want {
				return invalid(CodeInvalid, "derived_anchors.context")
			}
		}
	}
	return nil
}

func validAnchorReason(reason string) bool {
	switch reason {
	case "not_recorded", "not_provided", "input_limit", "asset_unavailable", "selector_unavailable", "anchor_language_unsupported", "anchor_parser_unavailable", "source_invalid", "source_missing", "parse_error", "anchor_ambiguous", "discarded":
		return true
	}
	return false
}

// ResolveFindingAnchors stamps verified Git ownership onto a validated Runtime
// derivation. The caller verifies selected Stage input and exact source bytes.
func ResolveFindingAnchors(basis FindingAnchorDerivation, repository, commit string, source contracts.ExactArtifact, proposal FindingProposal) (FindingAnchorProjection, error) {
	if err := validateAnchorDerivation(basis); err != nil {
		return FindingAnchorProjection{}, err
	}
	if basis.Availability == "unavailable" {
		return UnavailableFindingAnchors(basis.Reason), nil
	}
	if len(basis.Anchors) != len(proposal.Locations) {
		return FindingAnchorProjection{}, invalid(CodeInvalid, "derived_anchors.locations")
	}
	for i, anchor := range basis.Anchors {
		location := proposal.Locations[i]
		start, end := int64(0), int64(0)
		if location.Line != nil {
			start, end = *location.Line, *location.Line
		} else if location.Range != nil {
			start, end = location.Range.StartLine, location.Range.EndLine
		}
		if location.File != anchor.File || start != anchor.StartLine || end != anchor.EndLine {
			return FindingAnchorProjection{}, invalid(CodeInvalid, "derived_anchors.locations")
		}
		if anchor.Context != nil && !qualifiesLocalAnchor(proposal) {
			return FindingAnchorProjection{}, invalid(CodeInvalid, "derived_anchors.context")
		}
	}
	result := FindingAnchorProjection{Schema: FindingAnchorProjectionSchema, Availability: "available", Repository: repository, Commit: commit, Source: &source, Basis: &basis, Keys: anchorKeys(basis, repository, commit, source.Digest)}
	if _, err := EncodeFindingAnchors(result); err != nil {
		return FindingAnchorProjection{}, err
	}
	return result, nil
}

func qualifiesLocalAnchor(proposal FindingProposal) bool {
	return len(proposal.Locations) == 1 && len(proposal.StandardRefs) == 1 && proposal.StandardRefs[0].Scheme == "CWE" && proposal.StandardRefs[0].RequirementID == "CWE-321" && proposal.Subject == nil && proposal.Hypothesis == "" && len(proposal.Preconditions) == 0 && len(proposal.ProposedChecks) == 0 && len(proposal.Limitations) == 0 && proposal.HTTPExchange == nil
}

func anchorKeys(basis FindingAnchorDerivation, repository, commit, digest string) []FindingAnchorKeys {
	keys := make([]FindingAnchorKeys, len(basis.Anchors))
	for i, a := range basis.Anchors {
		site, _ := contentdigest.CanonicalJSON([]any{"contractor/structural-site/v1", []string{"git", repository}, a.File, a.Profile, a.Grammar, a.SignatureDigest, a.Path, a.NodeDigest})
		keys[i].SiteKey = strings.TrimPrefix(site, "sha256:")
		keys[i].IssueReason = "issue_context_unavailable"
		if c := a.Context; c != nil {
			issue, _ := contentdigest.CanonicalJSON([]any{"contractor/snapshot-issue-anchor/v1", []string{"git", repository}, []string{commit, digest}, c.RuleID, keys[i].SiteKey, c.Scheme, c.ContextDigest})
			keys[i].IssueKey = strings.TrimPrefix(issue, "sha256:")
			keys[i].IssueReason = ""
		}
	}
	return keys
}

func EncodeFindingAnchors(value FindingAnchorProjection) ([]byte, error) {
	if err := validateAnchorProjection(value); err != nil {
		return nil, err
	}
	data, err := strictjson.Canonical(value)
	if err != nil || len(data) > MaximumFindingAnchorProjectionBytes {
		return nil, invalid(CodeLimitExceeded, "anchors")
	}
	return data, nil
}

func DecodeFindingAnchors(data []byte) (FindingAnchorProjection, error) {
	var value FindingAnchorProjection
	if len(data) > MaximumFindingAnchorProjectionBytes {
		return value, invalid(CodeLimitExceeded, "anchors")
	}
	if err := strictjson.DecodeNonNull(data, &value); err != nil {
		return value, err
	}
	if err := validateAnchorProjection(value); err != nil {
		return value, err
	}
	var fields map[string]json.RawMessage
	if err := strictjson.Decode(data, &fields); err != nil {
		return value, err
	}
	if value.Availability == "unavailable" && len(fields) != 3 || value.Availability == "available" && len(fields) != 7 {
		return value, invalid(CodeInvalid, "anchors")
	}
	if err := exactAnchorShape(data, value); err != nil {
		return value, err
	}
	return value, nil
}

func validateAnchorProjection(value FindingAnchorProjection) error {
	if value.Schema != FindingAnchorProjectionSchema {
		return invalid(CodeInvalid, "anchors.schema")
	}
	if value.Availability == "unavailable" {
		if !validAnchorReason(value.Reason) || value.Source != nil || value.Basis != nil || len(value.Keys) != 0 || value.Repository != "" || value.Commit != "" {
			return invalid(CodeInvalid, "anchors")
		}
		return nil
	}
	if value.Availability != "available" || value.Reason != "" || !validFingerprintRepository(value.Repository) || !fingerprintCommit.MatchString(value.Commit) || value.Source == nil || value.Source.Validate() != nil || !value.Source.HasMetadata() || value.Source.MediaType != "application/zip" || value.Source.SizeBytes > MaximumFingerprintReportBytes || value.Basis == nil || validateAnchorDerivation(*value.Basis) != nil || value.Basis.Availability != "available" || !value.Source.Ref.SameExact(*value.Basis.SourceRef) || value.Source.Digest != value.Basis.SourceDigest || len(value.Keys) != len(value.Basis.Anchors) {
		return invalid(CodeInvalid, "anchors")
	}
	want := anchorKeys(*value.Basis, value.Repository, value.Commit, value.Source.Digest)
	for i, k := range value.Keys {
		if k != want[i] {
			return invalid(CodeInvalid, "anchors.keys")
		}
	}
	return nil
}

// Reject explicit empty optional fields as well as unknown, missing and null
// members. Comparing canonical forms permits ordinary whitespace/key order.
func exactAnchorShape(data []byte, value any) error {
	var raw any
	if err := strictjson.DecodeNonNull(data, &raw); err != nil {
		return err
	}
	original, err := strictjson.Canonical(raw)
	if err != nil {
		return err
	}
	encoded, err := strictjson.Canonical(value)
	if err != nil || !bytes.Equal(original, encoded) {
		return invalid(CodeInvalid, "anchors.shape")
	}
	return nil
}
