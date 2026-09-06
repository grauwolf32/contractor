-- A requirement-applicability review can explicitly remove one exact
-- requirement from the applicable denominator. This remains separate from
-- rejecting an action: not_applicable records a first-class coverage result
-- and requires the owner's durable rationale.

ALTER TABLE audit_review_decisions
    DROP CONSTRAINT audit_review_decisions_action_check,
    ADD CONSTRAINT audit_review_decisions_action_check CHECK (
        action IN (
            'true_positive', 'false_positive', 'duplicate', 'reopen',
            'needs_evidence', 'approve', 'reject', 'not_applicable'
        )
    ),
    DROP CONSTRAINT audit_review_decisions_shape,
    ADD CONSTRAINT audit_review_decisions_shape CHECK (
        (
            finding_id IS NOT NULL
            AND verdict IS NOT NULL
            AND action = verdict
            AND (verdict = 'true_positive') = (severity IS NOT NULL)
            AND (verdict = 'duplicate') = (duplicate_target_id IS NOT NULL)
        ) OR (
            finding_id IS NULL
            AND verdict IS NULL
            AND severity IS NULL
            AND duplicate_target_id IS NULL
            AND action IN ('approve', 'reject', 'not_applicable')
        )
    );
