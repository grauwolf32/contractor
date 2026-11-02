package pgstore

// materialSQL reads an accepted exact output by the indexed scoped address.
var materialSQL = `
SELECT artifact_digest,media_type,size_bytes FROM audit_artifact_links
WHERE audit_id=$1 AND artifact_scope='audit'
  AND artifact_ref->>'namespace'=$2 AND artifact_ref->>'name'=$3
  AND artifact_ref->>'revision'=$4
ORDER BY logical_key LIMIT 1`
