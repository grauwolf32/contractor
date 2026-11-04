package pgstore

// Candidate keys are limited before document hydration and byte accounting.
const hypothesisPageSQL = `
WITH candidates AS MATERIALIZED (
 SELECT h.hypothesis_id,h.source_key,h.fixed,h.ordinal,h.current_revision,h.current_bytes
 FROM audit_hypotheses h WHERE h.audit_id=$1 AND h.ordinal>$2
 ORDER BY h.ordinal LIMIT $3
), bounded AS MATERIALIZED (
 SELECT *,sum(1024+current_bytes) OVER(ORDER BY ordinal) AS used_bytes,
 count(*) OVER() AS candidate_count FROM candidates
)
SELECT r.document,b.source_key,b.fixed,b.ordinal,b.candidate_count
 FROM bounded b JOIN audit_hypothesis_revisions r ON r.hypothesis_id=b.hypothesis_id AND r.revision=b.current_revision
 WHERE used_bytes<=1048576 ORDER BY b.ordinal`

const exactHypothesisSQL = `SELECT r.document,h.source_key,h.fixed,h.ordinal
 FROM audit_hypotheses h JOIN audit_hypothesis_revisions r USING(hypothesis_id,audit_id)
 WHERE h.audit_id=$1 AND h.hypothesis_id=$2 AND r.revision=$3`

const hypothesisUsageSQL = `SELECT hypothesis_count,hypothesis_revision_count,hypothesis_bytes
 FROM audits WHERE audit_id=$1`

const updateHypothesisUsageSQL = `UPDATE audits SET hypothesis_count=$2,
 hypothesis_revision_count=$3,hypothesis_bytes=$4 WHERE audit_id=$1`

var hypothesisWriteEvents = auditEvents(auditEvent("'hypotheses.changed'", "audit.audit_id", "audit.revision", "jsonb_build_object('count',$2::integer)"))
var advanceHypothesesSQL = `UPDATE audits AS audit SET updated_at=clock_timestamp(),` + hypothesisWriteEvents.advance() + `
 WHERE audit_id=$1 RETURNING audit.revision,` + hypothesisWriteEvents.appended()
