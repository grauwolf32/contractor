package findingintake

// SQL statements for project_purge.go.

// purgeProjectReceiptBatchSQL deletes the first finding proposal receipts of
// Project $1 in receipt_id order: at most $2 of them, and only while the
// stored size of their report projections and evidence, summed in that order,
// stays within $3 bytes, except that the first receipt always goes. It reads
// at most $2 + 1 receipts and returns how many it deleted and how many it
// read; fewer deleted than read means receipts remain. pg_column_size reads
// the stored size without detoasting. Used by PurgeProjectReceipts.
var purgeProjectReceiptBatchSQL = `
WITH candidates AS (
    SELECT receipt_id,
           row_number() OVER (ORDER BY receipt_id) AS position,
           sum(pg_column_size(report_projection) + pg_column_size(evidence))
               OVER (ORDER BY receipt_id) AS stored_bytes
      FROM finding_proposal_receipts
     WHERE project_id = $1
     ORDER BY receipt_id
     LIMIT $2::integer + 1
), purged AS (
    DELETE FROM finding_proposal_receipts AS receipt
     USING candidates
     WHERE receipt.receipt_id = candidates.receipt_id
       AND candidates.position <= $2::integer
       AND (candidates.position = 1 OR candidates.stored_bytes <= $3::bigint)
    RETURNING receipt.receipt_id
)
SELECT (SELECT count(*) FROM purged), (SELECT count(*) FROM candidates)`
