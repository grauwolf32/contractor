package pgstore

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
)

var comparisonHex = regexp.MustCompile(`^[0-9a-f]{64}$`)

func validComparisonIDs(ids ...string) bool {
	for _, id := range ids {
		if !auditstore.ValidReviewIdentity(id, 256) {
			return false
		}
	}
	return true
}

// CorrelationFence reads current compact identities for zero to two selected
// Audits. Project revisions are used only for Project-wide traversals.
func (s *QueryStore) CorrelationFence(ctx context.Context, owner, project string, audits []string) (auditstore.FindingCorrelationFence, error) {
	result := auditstore.FindingCorrelationFence{ProjectID: project, Audits: []auditstore.FindingComparisonAudit{}}
	if !validComparisonIDs(owner, project) || len(audits) > 2 || !validComparisonIDs(audits...) || len(audits) == 2 && audits[0] == audits[1] {
		return result, auditstore.ErrInvalid
	}
	rows, err := s.db.Query(ctx, correlationFenceSQL, owner, project, audits)
	if err != nil {
		return result, fmt.Errorf("read finding correlation fence: %w", err)
	}
	defer rows.Close()
	found := false
	for rows.Next() {
		var projectRevision int64
		var auditID, state, name, version, digest, baseline *string
		var revision *int64
		if err := rows.Scan(&projectRevision, &auditID, &revision, &state, &name, &version, &digest, &baseline); err != nil {
			return result, err
		}
		if projectRevision < 1 || uint64(projectRevision) > auditdomain.MaximumFindingCorrelationRevision {
			return result, errors.New("invalid stored correlation revision")
		}
		result.ProjectRevision = uint64(projectRevision)
		found = true
		if auditID != nil {
			if revision == nil || *revision < 1 || uint64(*revision) > auditdomain.MaximumFindingCorrelationRevision || state == nil || name == nil || version == nil || digest == nil || !contentdigest.Valid(*digest) || !auditstore.AuditState(*state).Valid() {
				return result, errors.New("invalid stored comparison Audit")
			}
			result.Audits = append(result.Audits, auditstore.FindingComparisonAudit{AuditID: *auditID, Revision: uint64(*revision), State: auditstore.AuditState(*state), Profile: auditstore.FindingComparisonProfile{Name: *name, Version: *version, Digest: *digest}, BaselineDigest: baseline, BaselineDigestEncoding: "postgres-jsonb-text/v1", Coverage: "not-compared"})
		}
	}
	if err := rows.Err(); err != nil {
		return result, err
	}
	if !found || len(result.Audits) != len(audits) {
		return result, auditstore.ErrNotFound
	}
	return result, nil
}

// CandidateSource verifies the selected receipt belongs to this exact finding.
func (s *QueryStore) CandidateSource(ctx context.Context, owner, audit, finding, receipt string) (auditstore.FindingCandidateSource, error) {
	var result auditstore.FindingCandidateSource
	if !validComparisonIDs(owner, audit, finding) || receipt != "" && !validComparisonIDs(receipt) {
		return result, auditstore.ErrInvalid
	}
	var selected any
	if receipt != "" {
		selected = receipt
	}
	var raw []byte
	err := s.db.QueryRow(ctx, candidateSourceSQL, owner, audit, finding, selected).Scan(&result.FindingID, &result.AuditID, &result.ProjectID, &result.FirstReceiptID, &result.SelectedReceiptID, &result.GroupKey, &raw, &result.AuditRevision, &result.ProjectRevision)
	if errors.Is(err, pgx.ErrNoRows) {
		return result, auditstore.ErrNotFound
	}
	if err != nil {
		return result, err
	}
	value, err := decodeComparisonProjection(raw, true)
	if err != nil {
		return result, err
	}
	result.Hints, result.HintsReason = value.Hints, value.Reason
	return result, nil
}

func (s *QueryStore) authorizeCorrelationRead(ctx context.Context, owner, project string, audits []string) error {
	if !validComparisonIDs(owner, project) || len(audits) > 2 || !validComparisonIDs(audits...) {
		return auditstore.ErrInvalid
	}
	var allowed bool
	err := s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM projects WHERE owner_id=$1 AND project_id=$2)
 AND (SELECT count(*) FROM audits WHERE owner_id=$1 AND project_id=$2 AND audit_id=ANY($3::text[]))=cardinality($3::text[])`, owner, project, audits).Scan(&allowed)
	if err != nil {
		return err
	}
	if !allowed {
		return auditstore.ErrNotFound
	}
	return nil
}

// CandidateWindow returns only matching inputs, not hydrated result previews.
// A caller's continuation advances over examined rows even when none match.
func (s *QueryStore) CandidateWindow(ctx context.Context, p auditstore.FindingCandidateSeek) ([]auditstore.FindingCandidateRecord, error) {
	ids := []string{}
	if p.AuditID != "" {
		ids = append(ids, p.AuditID)
	}
	if err := s.authorizeCorrelationRead(ctx, p.OwnerID, p.ProjectID, ids); err != nil {
		return nil, err
	}
	if !validComparisonIDs(p.ExcludeFindingID) || (p.AfterCreatedAt == nil) != (p.AfterFindingID == "") || p.AfterFindingID != "" && !validComparisonIDs(p.AfterFindingID) {
		return nil, auditstore.ErrInvalid
	}
	query, scope := candidateProjectWindowSQL, p.ProjectID
	if p.AuditID != "" {
		query, scope = candidateAuditWindowSQL, p.AuditID
	}
	var after any = pgtype.Timestamptz{InfinityModifier: pgtype.NegativeInfinity, Valid: true}
	if p.AfterCreatedAt != nil {
		after = *p.AfterCreatedAt
	}
	args := []any{scope, p.ExcludeFindingID, after, p.AfterFindingID, auditstore.FindingCandidateWindowRows}
	if p.GroupKey != "" {
		if !comparisonHex.MatchString(p.GroupKey) {
			return nil, auditstore.ErrInvalid
		}
		query = candidateProjectGroupWindowSQL
		if p.AuditID != "" {
			query = candidateAuditGroupWindowSQL
		}
		args = append(args, p.GroupKey)
	}
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := make([]auditstore.FindingCandidateRecord, 0, auditstore.FindingCandidateWindowRows)
	for rows.Next() {
		var value auditstore.FindingCandidateRecord
		var raw []byte
		if err := rows.Scan(&value.FindingID, &value.AuditID, &value.FirstReceiptID, &value.GroupKey, &value.CreatedAt, &raw); err != nil {
			return nil, err
		}
		projection, err := decodeComparisonProjection(raw, false)
		if err != nil {
			return nil, err
		}
		value.Hints, value.HintsReason = projection.Hints, projection.Reason
		result = append(result, value)
	}
	return result, rows.Err()
}

// ComparisonGroups seeks distinct groups, or individual unmatched observations,
// in precisely the two selected Audits, with no Project-wide filtering scan.
func (s *QueryStore) ComparisonGroups(ctx context.Context, p auditstore.FindingComparisonSeek) ([]auditstore.FindingComparisonGroup, error) {
	if p.LeftAuditID == p.RightAuditID || p.Limit < 1 || p.Limit > auditstore.FindingComparisonMaximumLimit+1 {
		return nil, auditstore.ErrInvalid
	}
	if err := s.authorizeCorrelationRead(ctx, p.OwnerID, p.ProjectID, []string{p.LeftAuditID, p.RightAuditID}); err != nil {
		return nil, err
	}
	if p.AfterKey != "" && (!p.Unmatched && !comparisonHex.MatchString(p.AfterKey) || p.Unmatched && !validComparisonIDs(p.AfterKey)) {
		return nil, auditstore.ErrInvalid
	}
	query := comparisonGroupsSQL
	if p.Unmatched {
		query = comparisonUnmatchedSQL
	}
	rows, err := s.db.Query(ctx, query, p.LeftAuditID, p.RightAuditID, p.AfterKey, p.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := make([]auditstore.FindingComparisonGroup, 0, p.Limit)
	for rows.Next() {
		var value auditstore.FindingComparisonGroup
		if err := rows.Scan(&value.GroupKey, &value.Algorithm, &value.Value, &value.LeftFindingID, &value.RightFindingID, &value.UnavailableReason); err != nil {
			return nil, err
		}
		if value.LeftFindingID == nil && value.RightFindingID == nil {
			return nil, errors.New("empty comparison group")
		}
		result = append(result, value)
	}
	return result, rows.Err()
}

// CorrelationMembers returns one keyset page of retained member IDs in the
// declared Project, Audit or two-Audit scope.
func (s *QueryStore) CorrelationMembers(ctx context.Context, p auditstore.FindingMemberSeek) ([]auditdomain.FindingCorrelationReference, error) {
	if !comparisonHex.MatchString(p.GroupKey) || p.Limit < 1 || p.Limit > auditstore.FindingComparisonMaximumLimit+1 || p.AfterFindingID != "" && !validComparisonIDs(p.AfterFindingID) {
		return nil, auditstore.ErrInvalid
	}
	if err := s.authorizeCorrelationRead(ctx, p.OwnerID, p.ProjectID, p.AuditIDs); err != nil {
		return nil, err
	}
	query, args := projectMembersSQL, []any{p.ProjectID, p.GroupKey, p.AfterFindingID, p.Limit}
	if len(p.AuditIDs) > 0 {
		query = auditMembersSQL
		args[0] = p.AuditIDs[0]
	}
	if len(p.AuditIDs) == 2 {
		query = pairMembersSQL
		args = append(args, p.AuditIDs[1])
	}
	rows, err := s.db.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := make([]auditdomain.FindingCorrelationReference, 0, p.Limit)
	for rows.Next() {
		var id auditdomain.FindingCorrelationReference
		if err := rows.Scan(&id.FindingID, &id.AuditID); err != nil {
			return nil, err
		}
		result = append(result, id)
	}
	return result, rows.Err()
}

// ComparisonPreviews reads only the bounded display projection and exact
// current result references, in one owner-authorized batch without Artifact I/O.
func (s *QueryStore) ComparisonPreviews(ctx context.Context, owner, project string, ids []string) (map[string]auditstore.FindingComparisonPreview, error) {
	if !validComparisonIDs(owner, project) || len(ids) > 2*auditstore.FindingComparisonMaximumLimit || !validComparisonIDs(ids...) {
		return nil, auditstore.ErrInvalid
	}
	result := make(map[string]auditstore.FindingComparisonPreview, len(ids))
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, comparisonPreviewsSQL, owner, project, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var value auditstore.FindingComparisonPreview
		var raw []byte
		var revision int64
		if err := rows.Scan(&value.FindingID, &value.AuditID, &revision, &value.State, &raw, &value.AnalystVerdict, &value.AnalystSeverity, &value.SemanticAssessment, &value.AssessmentID, &value.DecisionID, &value.Origin.ReceiptID, &value.Origin.RunID, &value.Origin.RunDeleted, &value.Origin.LogicalAgentName, &value.Origin.WorkflowName, &value.Origin.WorkflowVersion, &value.Origin.WorkflowClosureDigest); err != nil {
			return nil, err
		}
		var preview struct {
			Title              string `json:"title" strictjson:"required"`
			TitleTruncated     bool   `json:"titleTruncated" strictjson:"required"`
			SeveritySuggestion string `json:"severitySuggestion" strictjson:"required"`
		}
		if len(raw) == 0 || string(raw) == "null" {
			return nil, auditstore.ErrCorrelationItemTooLarge
		}
		if err := strictjson.Decode(raw, &preview); err != nil {
			return nil, err
		}
		if revision < 1 || uint64(revision) > auditdomain.MaximumFindingCorrelationRevision || !value.State.Valid() || preview.Title == "" || len(preview.Title) > 1024 || utf8.RuneCountInString(preview.Title) > 256 || !utf8.ValidString(preview.Title) || preview.SeveritySuggestion != "" && !auditstore.FindingSeverity(preview.SeveritySuggestion).Valid() || !contentdigest.Valid(value.Origin.WorkflowClosureDigest) {
			return nil, errors.New("invalid stored comparison preview")
		}
		value.Preview = true
		value.Revision = uint64(revision)
		value.Title = preview.Title
		value.TitleTruncated = preview.TitleTruncated
		value.SeveritySuggestion = preview.SeveritySuggestion
		result[value.FindingID] = value
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(result) != len(ids) {
		return nil, auditstore.ErrNotFound
	}
	return result, nil
}

func decodeComparisonProjection(raw []byte, display bool) (auditstore.FindingComparisonProjection, error) {
	var value auditstore.FindingComparisonProjection
	if len(raw) > 32768 {
		return value, errors.New("oversized stored comparison projection")
	}
	if err := strictjson.Decode(raw, &value); err != nil {
		return value, err
	}
	if value.Schema != "contractor.finding.comparison-projection.v1" {
		return value, errors.New("unknown stored comparison projection")
	}
	if value.Availability == "unavailable" {
		if value.Reason != "not_recorded" && value.Reason != "discarded" || value.Hints != nil || value.Preview != nil {
			return value, errors.New("invalid unavailable comparison projection")
		}
		return value, nil
	}
	if value.Availability != "available" || value.Reason != "" || value.Hints == nil || display && value.Preview == nil || !display && value.Preview != nil {
		return value, errors.New("invalid stored comparison projection")
	}
	h := value.Hints
	if h.Families == nil || h.Sites == nil || len(h.Families) > 32 || len(h.Sites) > 32 || !h.FamiliesAvailable && len(h.Families) > 0 || h.ProducerKey != "" && !comparisonHex.MatchString(h.ProducerKey) || len(h.Repository) > 4096 || h.Commit != "" && (len(h.Commit) != 40 || strings.Trim(h.Commit, "0123456789abcdef") != "") {
		return value, errors.New("invalid stored comparison hints")
	}
	if h.Subject != nil && (!auditstore.ValidReviewIdentity(h.Subject.Kind, auditdomain.MaximumIdentifierBytes) || !auditstore.ValidReviewIdentity(h.Subject.Key, auditdomain.MaximumIdentifierBytes)) {
		return value, errors.New("invalid comparison subject")
	}
	for _, site := range h.Sites {
		if !comparisonHex.MatchString(site) {
			return value, errors.New("invalid comparison site")
		}
	}
	for _, ref := range h.Families {
		if (!auditstore.ValidReviewIdentity(ref.Scheme, auditdomain.MaximumIdentifierBytes) || !auditstore.ValidReviewIdentity(ref.RequirementID, auditdomain.MaximumIdentifierBytes)) || ref.Version == "" || len(ref.Version) > 160 || !utf8.ValidString(ref.Version) {
			return value, errors.New("invalid comparison family")
		}
	}
	if display && value.Preview != nil {
		if value.Preview.Title == "" || len(value.Preview.Title) > 1024 || utf8.RuneCountInString(value.Preview.Title) > 256 {
			return value, errors.New("invalid comparison title")
		}
	}
	return value, nil
}

// CheckCorrelationFence joins only scalar revisions after bounded hydration.
// A missing parent or changed result is stale after the initial authorization.
func (s *QueryStore) CheckCorrelationFence(ctx context.Context, owner string, before auditstore.FindingCorrelationFence) error {
	if !validComparisonIDs(owner, before.ProjectID) || len(before.Audits) > 2 {
		return auditstore.ErrInvalid
	}
	ids := make([]string, len(before.Audits))
	revisions := make([]int64, len(before.Audits))
	for i, audit := range before.Audits {
		if !validComparisonIDs(audit.AuditID) || audit.Revision == 0 || audit.Revision > auditdomain.MaximumFindingCorrelationRevision {
			return auditstore.ErrInvalid
		}
		ids[i] = audit.AuditID
		revisions[i] = int64(audit.Revision)
	}
	var equal bool
	if err := s.db.QueryRow(ctx, checkCorrelationFenceSQL, owner, before.ProjectID, int64(before.ProjectRevision), ids, revisions).Scan(&equal); err != nil {
		return err
	}
	if !equal {
		return auditstore.ErrCorrelationStale
	}
	return nil
}
