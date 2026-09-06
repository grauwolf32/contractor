package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// ListReportFindings returns the exact finding state selected by one report
// build. It is intentionally not an owner-facing query: the caller already
// holds the Audit Controller claim and freezes the returned projection into an
// immutable report artifact.
func (s *PostgresStore) ListReportFindings(
	ctx context.Context, auditID string,
) ([]ReportFinding, error) {
	if err := validateID("auditID", auditID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT finding.finding_id, finding.state, finding.first_proposal_ref,
       finding.duplicate_target_id, finding.revision,
       assessment.assessment_id, assessment.semantic_assessment,
       assessment.result_ref, assessment.result_digest,
       assessment.direct_verification, assessment.contract_ref,
       assessment.contract_digest, assessment.accepted_at,
       decision.decision_id, decision.actor_id, decision.verdict,
       decision.severity, decision.rationale, decision.subject_revision,
       decision.subject_digest, decision.created_at
  FROM audit_findings AS finding
  LEFT JOIN audit_finding_assessments AS assessment
    ON assessment.assessment_id = finding.current_assessment_id
  LEFT JOIN audit_review_decisions AS decision
    ON decision.decision_id = finding.current_decision_id
 WHERE finding.audit_id = $1
 ORDER BY finding.created_at, finding.finding_id
 LIMIT 100001`, auditID)
	if err != nil {
		return nil, fmt.Errorf("list Audit report findings: %w", err)
	}
	defer rows.Close()
	result := make([]ReportFinding, 0)
	for rows.Next() {
		var value ReportFinding
		var state string
		var proposalJSON []byte
		var revision int64
		var assessmentID, semantic, resultDigest, contractDigest *string
		var resultJSON, contractJSON []byte
		var direct *bool
		var assessmentAt *time.Time
		var decisionID, actor, verdict, severity, rationale, subjectDigest *string
		var subjectRevision *int64
		var decisionAt *time.Time
		if err := rows.Scan(
			&value.FindingID, &state, &proposalJSON, &value.DuplicateTargetID, &revision,
			&assessmentID, &semantic, &resultJSON, &resultDigest, &direct,
			&contractJSON, &contractDigest, &assessmentAt,
			&decisionID, &actor, &verdict, &severity, &rationale, &subjectRevision,
			&subjectDigest, &decisionAt,
		); err != nil {
			return nil, fmt.Errorf("scan Audit report finding: %w", err)
		}
		if revision < 1 || json.Unmarshal(proposalJSON, &value.FirstProposal) != nil ||
			validateExactArtifact("first proposal", value.FirstProposal, true) != nil {
			return nil, errors.New("stored Audit report finding is invalid")
		}
		value.State, value.Revision = state, uint64(revision)
		if assessmentID != nil {
			if semantic == nil || resultDigest == nil || direct == nil || assessmentAt == nil {
				return nil, errors.New("stored Audit report assessment is incomplete")
			}
			assessment := &ReportFindingAssessment{
				AssessmentID: *assessmentID, SemanticAssessment: *semantic,
				Result: ExactArtifact{Digest: *resultDigest}, DirectVerification: *direct,
				AcceptedAt: *assessmentAt,
			}
			if json.Unmarshal(resultJSON, &assessment.Result.Ref) != nil ||
				validateExactArtifact("assessment result", assessment.Result, false) != nil {
				return nil, errors.New("stored Audit report assessment result is invalid")
			}
			if contractJSON != nil {
				if contractDigest == nil {
					return nil, errors.New("stored Audit report assessment contract is incomplete")
				}
				assessment.Contract = &ExactArtifact{Digest: *contractDigest}
				if json.Unmarshal(contractJSON, &assessment.Contract.Ref) != nil ||
					validateExactArtifact("assessment contract", *assessment.Contract, false) != nil {
					return nil, errors.New("stored Audit report assessment contract is invalid")
				}
			}
			value.Assessment = assessment
		}
		if decisionID != nil {
			if actor == nil || verdict == nil || rationale == nil || subjectRevision == nil ||
				*subjectRevision < 1 || subjectDigest == nil || decisionAt == nil {
				return nil, errors.New("stored Audit report decision is incomplete")
			}
			value.Decision = &ReportFindingDecision{
				DecisionID: *decisionID, ActorID: *actor, Verdict: *verdict,
				Severity: severity, Rationale: *rationale,
				SubjectRevision: uint64(*subjectRevision), SubjectDigest: *subjectDigest,
				CreatedAt: *decisionAt,
			}
		}
		result = append(result, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Audit report findings: %w", err)
	}
	if len(result) > 100000 {
		return nil, errors.New("stored Audit finding set exceeds report bound")
	}
	return result, nil
}
