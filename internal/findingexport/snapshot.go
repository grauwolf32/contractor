package findingexport

import (
	"bytes"
	"encoding/json"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/documentnumber"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

var gitCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)

var ErrInvalid error = &auditdomain.ValidationError{Code: auditdomain.CodeInvalid, Field: "finding_export"}
var ErrLimit error = &auditdomain.ValidationError{Code: auditdomain.CodeLimitExceeded, Field: "finding_export"}

// Normalize binds request identity to one Audit; it never widens a selection.
func (r Request) Normalize(auditID string) (Request, string, error) {
	if contracts.ValidateResourceID("audit", auditID) != nil || !contracts.ValidName(r.ClientKey) || r.Format != "sarif" || r.Profile != "generic" || r.Findings == nil || len(r.Findings) > auditdomain.MaximumCollectionEntries {
		return Request{}, "", ErrInvalid
	}
	out := r
	out.Findings = append([]Selection{}, r.Findings...)
	sort.Slice(out.Findings, func(i, j int) bool { return out.Findings[i].FindingID < out.Findings[j].FindingID })
	for i, f := range out.Findings {
		if !contracts.ValidIdempotencyKey(f.FindingID) || f.Revision == 0 || f.Revision > contracts.MaxExactInteger || f.PresentationReceiptID != "" && !contracts.ValidIdempotencyKey(f.PresentationReceiptID) || i > 0 && out.Findings[i-1].FindingID == f.FindingID {
			return Request{}, "", ErrInvalid
		}
	}
	encoded, err := strictjson.Canonical([]any{auditID, out})
	if err != nil {
		return Request{}, "", ErrInvalid
	}
	return out, contentdigest.Bytes(encoded), nil
}

func NewSnapshot(auditID string, request Request, at time.Time) (Snapshot, error) {
	normalized, digest, err := request.Normalize(auditID)
	if err != nil {
		return Snapshot{}, err
	}
	value := Snapshot{Schema: SnapshotSchema, AuditID: auditID, SnapshotAt: at.UTC().Format(time.RFC3339Nano), ExporterVersion: ExporterVersion, PolicyVersion: PolicyVersion, Request: normalized, RequestDigest: digest, Receipts: []Receipt{}, Results: []Result{}}
	encoded, err := strictjson.Canonical(value)
	if err != nil {
		return Snapshot{}, ErrInvalid
	}
	value.size = len(encoded)
	return value, nil
}

// AddReceipt keeps only rendering inputs. Full outgoing requests, headers and
// bodies remain in the original access-controlled evidence, not export fields.
func (s *Snapshot) AddReceipt(value Receipt) error {
	proposal := value.Proposal
	proposal.HTTPExchange = nil
	proposal.Locations = append([]auditdomain.FindingLocation(nil), proposal.Locations...)
	for i := range proposal.Locations {
		if proposal.Locations[i].URL != "" {
			proposal.Locations[i].URL = SanitizedURI(proposal.Locations[i].URL)
		}
	}
	value.Proposal = proposal
	encoded, err := strictjson.Canonical(value)
	if err != nil {
		return ErrInvalid
	}
	extra := len(encoded)
	if len(s.Receipts) > 0 {
		extra++
	}
	if extra > MaxBytes-s.size {
		return ErrLimit
	}
	s.size += extra
	s.Receipts = append(s.Receipts, value)
	return nil
}

func (s *Snapshot) AddResult(value Result) error {
	value.ReceiptIDs = append([]string{}, value.ReceiptIDs...)
	sort.Strings(value.ReceiptIDs)
	encoded, err := strictjson.Canonical(value)
	if err != nil {
		return ErrInvalid
	}
	extra := len(encoded)
	if len(s.Results) > 0 {
		extra++
	}
	if extra > MaxBytes-s.size {
		return ErrLimit
	}
	s.size += extra
	s.Results = append(s.Results, value)
	return nil
}

func EncodeSnapshot(value Snapshot) ([]byte, error) {
	value.Receipts = append([]Receipt{}, value.Receipts...)
	value.Results = append([]Result{}, value.Results...)
	sort.Slice(value.Receipts, func(i, j int) bool { return value.Receipts[i].ReceiptID < value.Receipts[j].ReceiptID })
	sort.Slice(value.Results, func(i, j int) bool { return value.Results[i].FindingID < value.Results[j].FindingID })
	if err := validateSnapshot(value); err != nil {
		return nil, err
	}
	encoded, err := strictjson.Canonical(value)
	if err != nil {
		return nil, ErrInvalid
	}
	if len(encoded) > MaxBytes {
		return nil, ErrLimit
	}
	return encoded, nil
}

func DecodeSnapshot(data []byte) (Snapshot, error) {
	var value Snapshot
	if len(data) > MaxBytes {
		return value, ErrLimit
	}
	if !utf8.Valid(data) || !strictjson.ValidUnicodeEscapes(data) || strictjson.HasNULEscape(data) {
		return value, ErrInvalid
	}
	if _, err := strictjson.DecodeBounded(data, strictjson.Bounds{MaxDepth: 64, MaxNodes: 1000000, MaxStringBytes: auditdomain.MaximumStringBytes}, documentnumber.Float); err != nil {
		return value, ErrInvalid
	}
	if strictjson.DecodeNonNull(data, &value) != nil {
		return value, ErrInvalid
	}
	encoded, err := EncodeSnapshot(value)
	if err != nil || !bytes.Equal(data, encoded) {
		return Snapshot{}, ErrInvalid
	}
	return value, nil
}

func validateSnapshot(value Snapshot) error {
	normalized, digest, err := value.Request.Normalize(value.AuditID)
	if err != nil || value.Schema != SnapshotSchema || !contracts.ValidVersion(value.ExporterVersion) || value.PolicyVersion != PolicyVersion || digest != value.RequestDigest || !slices.Equal(normalized.Findings, value.Request.Findings) || value.Receipts == nil || len(value.Receipts) > auditdomain.MaximumCollectionEntries || value.Results == nil || len(value.Results) != len(value.Request.Findings) {
		return ErrInvalid
	}
	at, err := time.Parse(time.RFC3339Nano, value.SnapshotAt)
	if err != nil || at.UTC().Format(time.RFC3339Nano) != value.SnapshotAt {
		return ErrInvalid
	}
	receipts := map[string]bool{}
	for _, r := range value.Receipts {
		if !contracts.ValidIdempotencyKey(r.ReceiptID) || !contracts.ValidIdempotencyKey(r.ProposalID) || receipts[r.ReceiptID] || r.Evidence == nil || len(r.Evidence) > auditdomain.MaximumEvidencePerItem || r.Proposal.HTTPExchange != nil || len(r.Evidence) != len(r.Proposal.EvidenceIDs) {
			return ErrInvalid
		}
		receipts[r.ReceiptID] = true
		if _, err := auditdomain.EncodeFindingProposal(r.Proposal); err != nil {
			return ErrInvalid
		}
		if !contracts.ValidIdempotencyKey(r.Origin.RunID) || !contracts.ValidIdempotencyKey(r.Origin.InvocationID) || !contentdigest.Valid(r.Origin.WorkflowClosureDigest) {
			return ErrInvalid
		}
		if r.HTTP != nil && r.HTTP.Validate() != nil || r.Analysis != nil && (r.Analysis.Validate() != nil || r.Analysis.Analysis.ClientKey != r.Proposal.ClientKey) {
			return ErrInvalid
		}
		for _, e := range append([]Evidence{r.ProposalArtifact}, r.Evidence...) {
			for _, doc := range []auditdomain.FindingCollectionDocument{e.Original, e.Retained} {
				id, err := auditdomain.FindingCollectionDocumentID(doc)
				if err != nil || id != doc.ID {
					return ErrInvalid
				}
			}
			if e.Git != nil {
				u, err := url.Parse(e.Git.RepositoryURI)
				if err != nil || len(e.Git.RepositoryURI) > 4096 || u.Host == "" || (u.Scheme != "https" && u.Scheme != "ssh") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || (u.Scheme == "https" && u.User != nil) || !gitCommit.MatchString(e.Git.Commit) {
					return ErrInvalid
				}
				if u.User != nil {
					if _, password := u.User.Password(); password {
						return ErrInvalid
					}
				}
			}
			if e.Original.Digest != e.Retained.Digest || e.Original.SizeBytes != e.Retained.SizeBytes || e.Original.MediaType != e.Retained.MediaType {
				return ErrInvalid
			}
		}
	}
	used := map[string]bool{}
	for i, r := range value.Results {
		selection := value.Request.Findings[i]
		if r.FindingID != selection.FindingID || r.Revision != selection.Revision || len(r.ReceiptIDs) == 0 || len(r.ReceiptIDs) > auditdomain.MaximumCollectionEntries || !slices.IsSorted(r.ReceiptIDs) || !slices.Contains(r.ReceiptIDs, r.PresentationReceiptID) || selection.PresentationReceiptID != "" && r.PresentationReceiptID != selection.PresentationReceiptID {
			return ErrInvalid
		}
		for j, id := range r.ReceiptIDs {
			if !receipts[id] || j > 0 && r.ReceiptIDs[j-1] == id {
				return ErrInvalid
			}
			used[id] = true
		}
		if !slices.Contains([]string{"proposed", "confirmed", "rejected", "duplicate", "needs-evidence"}, r.Review.State) {
			return ErrInvalid
		}
		if (r.Review.State == "duplicate") != (r.Review.DuplicateTargetID != "") || r.Review.DuplicateTargetID == r.FindingID || r.Review.DuplicateTargetID != "" && !contracts.ValidIdempotencyKey(r.Review.DuplicateTargetID) {
			return ErrInvalid
		}
		if r.Review.Severity != "" && !slices.Contains([]string{"informational", "low", "medium", "high", "critical"}, r.Review.Severity) {
			return ErrInvalid
		}
		if r.Review.Severity != "" && (r.Review.State != "confirmed" || r.Review.Verdict != "true_positive" || r.Review.DecisionID == "") {
			return ErrInvalid
		}
		if p := r.PrimaryKey; p != nil {
			key, err := auditdomain.FindingGroupKey(p.ProjectID, p.Algorithm, p.Value)
			if err != nil || key != p.GroupKey || p.Policy == "" {
				return ErrInvalid
			}
		}
	}
	if len(used) != len(receipts) {
		return ErrInvalid
	}
	return nil
}

// SanitizedURI is the HTTP export projection, not a network destination.
func SanitizedURI(value string) string {
	parsed, err := url.Parse(value)
	if err != nil {
		return ""
	}
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	parsed.RawFragment = ""
	return parsed.String()
}

func EncodeDiagnostics(value Diagnostics) ([]byte, error) {
	if value.Schema != DiagnosticsSchema || contracts.ValidateResourceID("audit", value.AuditID) != nil || value.Items == nil || len(value.Items) > 4096 {
		return nil, ErrInvalid
	}
	for _, d := range value.Items {
		if !contracts.ValidName(d.Code) || d.Severity != "warning" || !contracts.ValidIdempotencyKey(d.FindingID) || len(d.Field) > 256 || strings.TrimSpace(d.Field) == "" {
			return nil, ErrInvalid
		}
	}
	data, err := strictjson.Canonical(value)
	if err != nil {
		return nil, ErrInvalid
	}
	if len(data) > MaxDiagnosticsBytes {
		return nil, ErrLimit
	}
	return data, nil
}

func DecodeDiagnostics(data []byte) (Diagnostics, error) {
	var value Diagnostics
	if len(data) > MaxDiagnosticsBytes {
		return value, ErrLimit
	}
	if strictjson.DecodeNonNull(data, &value) != nil {
		return value, ErrInvalid
	}
	encoded, err := EncodeDiagnostics(value)
	if err != nil || !bytes.Equal(data, encoded) {
		return Diagnostics{}, ErrInvalid
	}
	return value, nil
}

func (r *Request) UnmarshalJSON(data []byte) error {
	type plain Request
	if err := strictjson.DecodeNonNull(data, (*plain)(r)); err != nil {
		return ErrInvalid
	}
	return nil
}

func (s *Selection) UnmarshalJSON(data []byte) error {
	type plain Selection
	var value plain
	if err := strictjson.DecodeNonNull(data, &value); err != nil {
		return ErrInvalid
	}
	var fields map[string]json.RawMessage
	if strictjson.Decode(data, &fields) != nil {
		return ErrInvalid
	}
	if _, present := fields["presentationReceiptId"]; present && value.PresentationReceiptID == "" {
		return ErrInvalid
	}
	*s = Selection(value)
	return nil
}
