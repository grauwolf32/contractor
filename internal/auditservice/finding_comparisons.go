package auditservice

import (
	"context"
	"encoding/json"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

const FindingCorrelationMaximumCursorBytes = 1024

type FindingCorrelationPosition struct {
	ProjectRevision uint64   `json:"p,omitempty"`
	AuditRevisions  []uint64 `json:"a,omitempty"`
	Stage           int      `json:"s"`
	Key             string   `json:"k,omitempty"`
	CreatedAt       string   `json:"t,omitempty"`
}

type FindingCorrelationPage struct {
	HasMore    bool    `json:"hasMore"`
	NextCursor *string `json:"nextCursor,omitempty"`
}
type FindingCorrelationScope struct {
	Kind      string   `json:"kind"`
	ProjectID string   `json:"projectId"`
	AuditIDs  []string `json:"auditIds"`
}

type FindingCandidateReason struct {
	Kind            string   `json:"kind"`
	Key             string   `json:"key"`
	SourceReceiptID string   `json:"sourceReceiptId"`
	TargetReceiptID string   `json:"targetReceiptId"`
	SourceSelector  *int     `json:"sourceSelector,omitempty"`
	TargetSelector  *int     `json:"targetSelector,omitempty"`
	Limitations     []string `json:"limitations"`
}

type FindingCandidate struct {
	Tier                    int                                 `json:"tier"`
	AutomaticallyCorrelated bool                                `json:"automaticallyCorrelated"`
	Finding                 auditstore.FindingComparisonPreview `json:"finding"`
	Reasons                 []FindingCandidateReason            `json:"reasons"`
	ReasonsTruncated        bool                                `json:"reasonsTruncated"`
}

type FindingCandidatesParams struct {
	OwnerID, AuditID, FindingID, Scope, TargetAuditID, SourceReceiptID string
	Limit                                                              int
	Cursor                                                             *FindingCorrelationPosition
}

type FindingCandidateSourceProjection struct {
	AuditID           string `json:"auditId"`
	FindingID         string `json:"findingId"`
	ReceiptID         string `json:"receiptId"`
	FirstReceiptID    string `json:"firstReceiptId"`
	HintsAvailability string `json:"hintsAvailability"`
	HintsReason       string `json:"hintsReason,omitempty"`
	FamiliesAvailable bool   `json:"familiesAvailable"`
}

type FindingCandidatesPage struct {
	PolicyVersion string                             `json:"policyVersion"`
	Scope         FindingCorrelationScope            `json:"scope"`
	Fence         auditstore.FindingCorrelationFence `json:"fence"`
	Source        FindingCandidateSourceProjection   `json:"source"`
	TargetBasis   string                             `json:"targetBasis"`
	Items         []FindingCandidate                 `json:"items"`
	RowsExamined  int                                `json:"rowsExamined"`
	WorkLimited   bool                               `json:"workLimited"`
	Page          FindingCorrelationPage             `json:"page"`
	Next          *FindingCorrelationPosition        `json:"-"`
}

type FindingComparisonParams struct {
	OwnerID, ProjectID, LeftAuditID, RightAuditID string
	Limit                                         int
	Cursor                                        *FindingCorrelationPosition
}
type FindingComparisonRow struct {
	Kind                 string                                  `json:"kind"`
	RowKey               string                                  `json:"rowKey"`
	GroupKey             string                                  `json:"groupKey,omitempty"`
	Basis                *auditdomain.FindingCorrelationBasis    `json:"basis,omitempty"`
	UnavailableReason    string                                  `json:"unavailableReason,omitempty"`
	Representative       auditdomain.FindingCorrelationReference `json:"representative"`
	Left                 *auditstore.FindingComparisonPreview    `json:"left,omitempty"`
	Right                *auditstore.FindingComparisonPreview    `json:"right,omitempty"`
	PreviewsShareReceipt bool                                    `json:"previewsShareReceipt"`
}

type FindingComparisonPage struct {
	PolicyVersion string                             `json:"policyVersion"`
	Scope         FindingCorrelationScope            `json:"scope"`
	Fence         auditstore.FindingCorrelationFence `json:"fence"`
	LeftAuditID   string                             `json:"leftAuditId"`
	RightAuditID  string                             `json:"rightAuditId"`
	Limitations   []string                           `json:"limitations"`
	Items         []FindingComparisonRow             `json:"items"`
	Page          FindingCorrelationPage             `json:"page"`
	Next          *FindingCorrelationPosition        `json:"-"`
}

type FindingMembersParams struct {
	OwnerID, ProjectID, GroupKey string
	AuditIDs                     []string
	Limit                        int
	Cursor                       *FindingCorrelationPosition
}
type FindingMembersPage struct {
	PolicyVersion  string                                  `json:"policyVersion"`
	Scope          FindingCorrelationScope                 `json:"scope"`
	Fence          auditstore.FindingCorrelationFence      `json:"fence"`
	GroupKey       string                                  `json:"groupKey"`
	Representative auditdomain.FindingCorrelationReference `json:"representative"`
	Items          []auditstore.FindingComparisonPreview   `json:"items"`
	Page           FindingCorrelationPage                  `json:"page"`
	Next           *FindingCorrelationPosition             `json:"-"`
}

func validCorrelationLimit(limit int) bool {
	return limit >= 1 && limit <= auditstore.FindingComparisonMaximumLimit
}
func correlationScope(project string, audits []string) FindingCorrelationScope {
	kind := "project"
	if len(audits) == 1 {
		kind = "audit"
	}
	if len(audits) == 2 {
		kind = "audit-pair"
	}
	return FindingCorrelationScope{Kind: kind, ProjectID: project, AuditIDs: append([]string{}, audits...)}
}
func uniqueCorrelationAudits(ids ...string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, id := range ids {
		if id != "" && !seen[id] {
			out = append(out, id)
			seen[id] = true
		}
	}
	sort.Strings(out)
	return out
}
func fenceRevisions(f auditstore.FindingCorrelationFence) []uint64 {
	out := make([]uint64, len(f.Audits))
	for i, a := range f.Audits {
		out[i] = a.Revision
	}
	return out
}
func (s *Service) loadCorrelationFence(ctx context.Context, owner, project string, audits []string, projectWide bool) (auditstore.FindingCorrelationFence, error) {
	fence, err := s.reads().CorrelationFence(ctx, owner, project, audits)
	if !projectWide {
		fence.ProjectRevision = 0
	}
	return fence, err
}
func (s *Service) beginCorrelationRead(ctx context.Context, owner, project string, audits []string, projectWide bool, cursor *FindingCorrelationPosition) (auditstore.FindingCorrelationFence, error) {
	fence, err := s.loadCorrelationFence(ctx, owner, project, audits, projectWide)
	if err != nil {
		return fence, err
	}
	if cursor != nil {
		if cursor.ProjectRevision != fence.ProjectRevision || len(cursor.AuditRevisions) != len(fence.Audits) {
			return fence, auditstore.ErrCorrelationStale
		}
		for i, revision := range cursor.AuditRevisions {
			if revision != fence.Audits[i].Revision {
				return fence, auditstore.ErrCorrelationStale
			}
		}
	}
	return fence, nil
}
func (s *Service) finishCorrelationRead(ctx context.Context, owner string, before auditstore.FindingCorrelationFence, readErr error) error {
	if err := s.reads().CheckCorrelationFence(ctx, owner, before); err != nil {
		return err
	}
	return readErr
}
func correlationPosition(f auditstore.FindingCorrelationFence, stage int, key, created string) *FindingCorrelationPosition {
	return &FindingCorrelationPosition{ProjectRevision: f.ProjectRevision, AuditRevisions: fenceRevisions(f), Stage: stage, Key: key, CreatedAt: created}
}
func fitsCorrelationPage(value any) (bool, error) {
	data, err := json.Marshal(value)
	return len(data)+FindingCorrelationMaximumCursorBytes+128 <= auditstore.FindingComparisonMaximumBytes, err
}

// FindingCandidates searches fixed deterministic tiers. Exact group tiers seek
// directly; weak tiers expose bounded indexed work progress, including empties.
func (s *Service) FindingCandidates(ctx context.Context, p FindingCandidatesParams) (FindingCandidatesPage, error) {
	out := FindingCandidatesPage{PolicyVersion: auditdomain.FindingCorrelationPolicy, TargetBasis: "immutable-first-receipt", Items: []FindingCandidate{}}
	if !validCorrelationLimit(p.Limit) || p.Scope != "project" && p.Scope != "audit" || p.Scope == "audit" && p.TargetAuditID != "" && p.TargetAuditID != p.AuditID {
		return out, auditstore.ErrInvalid
	}
	source, err := s.reads().CandidateSource(ctx, p.OwnerID, p.AuditID, p.FindingID, p.SourceReceiptID)
	if err != nil {
		return out, err
	}
	target := p.TargetAuditID
	if p.Scope == "audit" {
		target = p.AuditID
	}
	audits := uniqueCorrelationAudits(p.AuditID, target)
	projectWide := target == ""
	out.Fence, err = s.beginCorrelationRead(ctx, p.OwnerID, source.ProjectID, audits, projectWide, p.Cursor)
	if err != nil {
		return out, err
	}
	if projectWide && source.ProjectRevision != out.Fence.ProjectRevision {
		return out, auditstore.ErrCorrelationStale
	}
	sourceRevisionMatches := false
	for _, audit := range out.Fence.Audits {
		if audit.AuditID == source.AuditID {
			sourceRevisionMatches = source.AuditRevision == audit.Revision
		}
	}
	if !sourceRevisionMatches {
		return out, auditstore.ErrCorrelationStale
	}
	scopeAudits := []string{}
	if target != "" {
		scopeAudits = append(scopeAudits, target)
	}
	out.Scope = correlationScope(source.ProjectID, scopeAudits)
	out.Source = FindingCandidateSourceProjection{AuditID: source.AuditID, FindingID: source.FindingID, ReceiptID: source.SelectedReceiptID, FirstReceiptID: source.FirstReceiptID, HintsAvailability: "unavailable", HintsReason: source.HintsReason}
	if source.Hints != nil {
		out.Source.HintsAvailability = "available"
		out.Source.FamiliesAvailable = source.Hints.FamiliesAvailable
	}
	stage := 1
	var after *time.Time
	afterID := ""
	if p.Cursor != nil {
		stage = p.Cursor.Stage
		afterID = p.Cursor.Key
		if stage < 1 || stage > 4 || (p.Cursor.CreatedAt == "") != (afterID == "") {
			return out, auditstore.ErrInvalid
		}
		if p.Cursor.CreatedAt != "" {
			parsed, err := time.Parse(time.RFC3339Nano, p.Cursor.CreatedAt)
			if err != nil {
				return out, auditstore.ErrInvalid
			}
			after = &parsed
		}
	}
	group := ""
	for stage <= 4 {
		switch stage {
		case 1:
			group = source.GroupKey
		case 2:
			if source.Hints != nil && source.Hints.ProducerKey != "" {
				group, err = auditdomain.FindingGroupKey(source.ProjectID, "contractor/producer-issue/v1", source.Hints.ProducerKey)
				if err != nil {
					return out, err
				}
				if group == source.GroupKey {
					group = ""
				}
			}
		case 3:
			if source.Hints != nil && len(source.Hints.Sites) > 0 {
				group = ""
				break
			}
		case 4:
			if source.Hints != nil && source.Hints.Subject != nil && len(source.Hints.Families) > 0 {
				group = ""
				break
			}
		}
		applicable := stage <= 2 && group != "" || stage == 3 && source.Hints != nil && len(source.Hints.Sites) > 0 || stage == 4 && source.Hints != nil && source.Hints.Subject != nil && len(source.Hints.Families) > 0
		if applicable {
			break
		}
		stage++
		after, afterID = nil, ""
	}
	if stage > 4 {
		return out, s.finishCorrelationRead(ctx, p.OwnerID, out.Fence, nil)
	}
	window, err := s.reads().CandidateWindow(ctx, auditstore.FindingCandidateSeek{OwnerID: p.OwnerID, ProjectID: source.ProjectID, AuditID: target, ExcludeFindingID: p.FindingID, AfterCreatedAt: after, AfterFindingID: afterID, GroupKey: group})
	if err != nil {
		return out, err
	}
	out.RowsExamined = len(window)
	out.WorkLimited = len(window) == auditstore.FindingCandidateWindowRows
	type chosen struct {
		record    auditstore.FindingCandidateRecord
		reasons   []FindingCandidateReason
		truncated bool
	}
	matches := []chosen{}
	positions := []auditstore.FindingCandidateRecord{}
	var last *auditstore.FindingCandidateRecord
	for i := range window {
		row := window[i]
		last = &window[i]
		tier, reasons, truncated := candidateReasons(source, row)
		if tier != stage {
			continue
		}
		matches = append(matches, chosen{row, reasons, truncated})
		positions = append(positions, row)
		if len(matches) == p.Limit {
			break
		}
	}
	ids := make([]string, len(matches))
	for i, m := range matches {
		ids[i] = m.record.FindingID
	}
	if len(window)+len(ids)+3 > auditstore.FindingComparisonWorkRows {
		return out, auditstore.ErrInvalid
	}
	previews, err := s.reads().ComparisonPreviews(ctx, p.OwnerID, source.ProjectID, ids)
	if err := s.finishCorrelationRead(ctx, p.OwnerID, out.Fence, err); err != nil {
		return out, err
	}
	for _, m := range matches {
		out.Items = append(out.Items, FindingCandidate{Tier: stage, AutomaticallyCorrelated: stage == 1, Finding: previews[m.record.FindingID], Reasons: m.reasons, ReasonsTruncated: m.truncated})
	}
	initial := len(out.Items)
	for {
		fits, err := fitsCorrelationPage(out)
		if err != nil {
			return out, err
		}
		if fits {
			break
		}
		if len(out.Items) == 0 {
			return out, auditstore.ErrCorrelationItemTooLarge
		}
		out.Items = out.Items[:len(out.Items)-1]
	}
	if initial > 0 && len(out.Items) == 0 {
		return out, auditstore.ErrCorrelationItemTooLarge
	}
	if len(out.Items) < initial {
		row := positions[len(out.Items)-1]
		last = &row
	}
	consumedAll := last == nil || len(out.Items) == initial && last.FindingID == window[len(window)-1].FindingID
	if !consumedAll || len(window) == auditstore.FindingCandidateWindowRows {
		if last != nil {
			out.Next = correlationPosition(out.Fence, stage, last.FindingID, last.CreatedAt.UTC().Format(time.RFC3339Nano))
		}
	} else if stage < 4 {
		out.Next = correlationPosition(out.Fence, stage+1, "", "")
	}
	out.Page.HasMore = out.Next != nil
	return out, nil
}

func candidateReasons(source auditstore.FindingCandidateSource, target auditstore.FindingCandidateRecord) (int, []FindingCandidateReason, bool) {
	reason := func(kind, key, receipt string) FindingCandidateReason {
		return FindingCandidateReason{Kind: kind, Key: key, SourceReceiptID: receipt, TargetReceiptID: target.FirstReceiptID, Limitations: []string{}}
	}
	if source.GroupKey != "" && source.GroupKey == target.GroupKey {
		return 1, []FindingCandidateReason{reason("primary-equality", source.GroupKey, source.FirstReceiptID)}, false
	}
	if source.Hints == nil || target.Hints == nil {
		return 0, nil, false
	}
	a, b := source.Hints, target.Hints
	if a.ProducerKey != "" && a.ProducerKey == b.ProducerKey {
		r := reason("selected-producer-equality", a.ProducerKey, source.SelectedReceiptID)
		r.Limitations = []string{"secondary-key-does-not-regroup"}
		return 2, []FindingCandidateReason{r}, false
	}
	reasons := []FindingCandidateReason{}
	for i, left := range a.Sites {
		for j, right := range b.Sites {
			if left != right {
				continue
			}
			if len(reasons) == auditstore.FindingCandidateMaximumReasons {
				return 3, reasons, true
			}
			r := reason("structural-site", left, source.SelectedReceiptID)
			si, ti := i, j
			r.SourceSelector, r.TargetSelector = &si, &ti
			r.Limitations = []string{"site-is-not-issue-equality", "snapshot-and-context-not-proven"}
			reasons = append(reasons, r)
		}
	}
	if len(reasons) > 0 {
		return 3, reasons, false
	}
	if a.Subject == nil || b.Subject == nil || *a.Subject != *b.Subject {
		return 0, nil, false
	}
	for _, left := range a.Families {
		for _, right := range b.Families {
			if left != right {
				continue
			}
			if len(reasons) == auditstore.FindingCandidateMaximumReasons {
				return 4, reasons, true
			}
			key, _ := json.Marshal([]string{left.Scheme, left.Version, left.RequirementID})
			r := reason("project-subject-family", string(key), source.SelectedReceiptID)
			r.Limitations = []string{"project-hint-only", "asset-and-issue-context-not-proven"}
			reasons = append(reasons, r)
		}
	}
	if len(reasons) > 0 {
		return 4, reasons, false
	}
	return 0, nil, false
}

// CompareFindings traverses exact groups and then individual unmatched rows.
// Representatives belong to the selected pair, never an unrelated third Audit.
func (s *Service) CompareFindings(ctx context.Context, p FindingComparisonParams) (FindingComparisonPage, error) {
	out := FindingComparisonPage{PolicyVersion: auditdomain.FindingCorrelationPolicy, LeftAuditID: p.LeftAuditID, RightAuditID: p.RightAuditID, Limitations: []string{"absence-is-not-remediation", "coverage-not-compared", "qualified-snapshot-keys-only", "previews-are-not-group-totals"}, Items: []FindingComparisonRow{}}
	if !validCorrelationLimit(p.Limit) || p.LeftAuditID == p.RightAuditID {
		return out, auditstore.ErrInvalid
	}
	audits := uniqueCorrelationAudits(p.LeftAuditID, p.RightAuditID)
	if len(audits) != 2 {
		return out, auditstore.ErrInvalid
	}
	out.Scope = correlationScope(p.ProjectID, audits)
	var err error
	out.Fence, err = s.beginCorrelationRead(ctx, p.OwnerID, p.ProjectID, audits, false, p.Cursor)
	if err != nil {
		return out, err
	}
	stage, key := 1, ""
	if p.Cursor != nil {
		stage, key = p.Cursor.Stage, p.Cursor.Key
		if stage < 1 || stage > 2 || p.Cursor.CreatedAt != "" {
			return out, auditstore.ErrInvalid
		}
	}
	seek := auditstore.FindingComparisonSeek{OwnerID: p.OwnerID, ProjectID: p.ProjectID, LeftAuditID: p.LeftAuditID, RightAuditID: p.RightAuditID, Unmatched: stage == 2, AfterKey: key, Limit: p.Limit + 1}
	groups, err := s.reads().ComparisonGroups(ctx, seek)
	if err != nil {
		return out, err
	}
	if len(groups) == 0 && stage == 1 {
		stage = 2
		seek.Unmatched, seek.AfterKey = true, ""
		groups, err = s.reads().ComparisonGroups(ctx, seek)
		if err != nil {
			return out, err
		}
	}
	more := len(groups) > p.Limit
	if more {
		groups = groups[:p.Limit]
	}
	ids := []string{}
	for _, g := range groups {
		if g.LeftFindingID != nil {
			ids = append(ids, *g.LeftFindingID)
		}
		if g.RightFindingID != nil {
			ids = append(ids, *g.RightFindingID)
		}
	}
	previews, err := s.reads().ComparisonPreviews(ctx, p.OwnerID, p.ProjectID, ids)
	if err := s.finishCorrelationRead(ctx, p.OwnerID, out.Fence, err); err != nil {
		return out, err
	}
	for _, g := range groups {
		row := FindingComparisonRow{GroupKey: g.GroupKey, RowKey: "group:" + g.GroupKey, Kind: "both", UnavailableReason: g.UnavailableReason}
		if g.GroupKey != "" {
			row.Basis = &auditdomain.FindingCorrelationBasis{Algorithm: g.Algorithm, Value: g.Value}
		}
		if g.LeftFindingID != nil {
			v := previews[*g.LeftFindingID]
			row.Left = &v
			row.Representative = auditdomain.FindingCorrelationReference{AuditID: v.AuditID, FindingID: v.FindingID}
		} else {
			row.Kind = "right-only"
		}
		if g.RightFindingID != nil {
			v := previews[*g.RightFindingID]
			row.Right = &v
			if row.Left == nil || v.FindingID < row.Left.FindingID {
				row.Representative = auditdomain.FindingCorrelationReference{AuditID: v.AuditID, FindingID: v.FindingID}
			}
		} else {
			row.Kind = "left-only"
		}
		if g.GroupKey == "" {
			row.Kind = "unmatched"
			row.RowKey = "unmatched:" + row.Representative.FindingID
		}
		if row.GroupKey != "" {
			setComparisonRole(row.Left, row.Representative)
			setComparisonRole(row.Right, row.Representative)
		}
		row.PreviewsShareReceipt = row.Left != nil && row.Right != nil && row.Left.Origin.ReceiptID == row.Right.Origin.ReceiptID
		out.Items = append(out.Items, row)
	}
	initial := len(out.Items)
	for {
		fits, err := fitsCorrelationPage(out)
		if err != nil {
			return out, err
		}
		if fits {
			break
		}
		if len(out.Items) == 0 {
			return out, auditstore.ErrCorrelationItemTooLarge
		}
		out.Items = out.Items[:len(out.Items)-1]
	}
	if initial > 0 && len(out.Items) == 0 {
		return out, auditstore.ErrCorrelationItemTooLarge
	}
	if len(out.Items) > 0 {
		last := out.Items[len(out.Items)-1]
		lastKey := last.GroupKey
		if stage == 2 {
			lastKey = last.Representative.FindingID
		}
		if more || len(out.Items) < initial {
			out.Next = correlationPosition(out.Fence, stage, lastKey, "")
		} else if stage == 1 {
			out.Next = correlationPosition(out.Fence, 2, "", "")
		}
	}
	out.Page.HasMore = out.Next != nil
	return out, nil
}

// FindingGroupMembers pages every retained observation under the declared
// Project/Audit/pair fence, preserving independent results and receipt origins.
func (s *Service) FindingGroupMembers(ctx context.Context, p FindingMembersParams) (FindingMembersPage, error) {
	out := FindingMembersPage{PolicyVersion: auditdomain.FindingCorrelationPolicy, GroupKey: p.GroupKey, Items: []auditstore.FindingComparisonPreview{}}
	if !validCorrelationLimit(p.Limit) || len(p.AuditIDs) > 2 {
		return out, auditstore.ErrInvalid
	}
	audits := uniqueCorrelationAudits(p.AuditIDs...)
	if len(audits) != len(p.AuditIDs) {
		return out, auditstore.ErrInvalid
	}
	out.Scope = correlationScope(p.ProjectID, audits)
	projectWide := len(audits) == 0
	var err error
	out.Fence, err = s.beginCorrelationRead(ctx, p.OwnerID, p.ProjectID, audits, projectWide, p.Cursor)
	if err != nil {
		return out, err
	}
	after := ""
	if p.Cursor != nil {
		if p.Cursor.Stage != 1 || p.Cursor.Key == "" || p.Cursor.CreatedAt != "" {
			return out, auditstore.ErrInvalid
		}
		after = p.Cursor.Key
	}
	seek := auditstore.FindingMemberSeek{OwnerID: p.OwnerID, ProjectID: p.ProjectID, GroupKey: p.GroupKey, AuditIDs: audits, Limit: 2}
	first, err := s.reads().CorrelationMembers(ctx, seek)
	if err != nil {
		return out, err
	}
	if len(first) == 0 {
		return out, auditstore.ErrNotFound
	}
	seek.AfterFindingID, seek.Limit = after, p.Limit+1
	members, err := s.reads().CorrelationMembers(ctx, seek)
	if err != nil {
		return out, err
	}
	more := len(members) > p.Limit
	if more {
		members = members[:p.Limit]
	}
	ids := make([]string, len(members))
	for i, member := range members {
		ids[i] = member.FindingID
	}
	previews, err := s.reads().ComparisonPreviews(ctx, p.OwnerID, p.ProjectID, ids)
	if err := s.finishCorrelationRead(ctx, p.OwnerID, out.Fence, err); err != nil {
		return out, err
	}
	// The representative's Audit is selected by the same scoped first probe;
	// a one-row preview lookup is unnecessary on later pages.
	out.Representative = first[0]

	for _, id := range ids {
		value := previews[id]
		setComparisonRole(&value, out.Representative)
		out.Items = append(out.Items, value)
	}
	initial := len(out.Items)
	for {
		fits, err := fitsCorrelationPage(out)
		if err != nil {
			return out, err
		}
		if fits {
			break
		}
		if len(out.Items) == 0 {
			return out, auditstore.ErrCorrelationItemTooLarge
		}
		out.Items = out.Items[:len(out.Items)-1]
	}
	if initial > 0 && len(out.Items) == 0 {
		return out, auditstore.ErrCorrelationItemTooLarge
	}
	if more || len(out.Items) < initial {
		out.Next = correlationPosition(out.Fence, 1, out.Items[len(out.Items)-1].FindingID, "")
	}
	out.Page.HasMore = out.Next != nil
	return out, s.finishCorrelationRead(ctx, p.OwnerID, out.Fence, nil)
}

func setComparisonRole(value *auditstore.FindingComparisonPreview, representative auditdomain.FindingCorrelationReference) {
	if value == nil {
		return
	}
	value.CorrelationRole = "automatic-duplicate"
	if value.FindingID == representative.FindingID && value.AuditID == representative.AuditID {
		value.CorrelationRole = "representative"
	}
}
