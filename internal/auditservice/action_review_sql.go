package auditservice

// SQL statements for action_review.go.

// rejectAwaitingItemSQL settles Audit $1's item $2 as excluded when it is still
// awaiting review, and records the owner's rejection on its coverage: status
// excluded, the human-approval-rejected gap (added once) and a fixed
// rationale. The caller requires one affected row.
// Used by validateAndApplyItemDecision.
var rejectAwaitingItemSQL = `
UPDATE audit_items
   SET state = 'settled', final_disposition = 'excluded',
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond'),
       coverage_status = 'excluded',
       coverage_gaps = CASE WHEN coverage_gaps ? 'human-approval-rejected' THEN coverage_gaps
                            ELSE coverage_gaps || '["human-approval-rejected"]'::jsonb END,
       coverage_rationale = 'The exact proposed action was rejected by its owner.',
       coverage_updated_at = GREATEST(clock_timestamp(), coverage_updated_at + interval '1 microsecond')
 WHERE audit_id = $1 AND item_id = $2 AND state = 'awaiting_review'`
