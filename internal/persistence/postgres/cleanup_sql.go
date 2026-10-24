package postgres

// SQL statements for cleanup.go.

// boundedDeleteSQL deletes the longest prefix of one table's matching rows
// that fits the rest of a cleanup pass's budget (DeleteWithinBudget). %[1]s is
// the table and %[2]s one of the candidate statements below, which reads at
// most $1 + 1 matching rows, charges each row and numbers the rows in the
// order they are taken. $1 is the row mutations the pass may still make, $2
// the bytes it may still charge, $3 whether it has charged nothing yet and $4
// the row mutations one deleted row costs. A row joins while the running row
// cost stays within $1 and the running charge within $2; the first row of a
// fresh pass joins whatever its charge. Every candidate is charged before any
// is deleted, and a row a concurrent writer replaced since the read is left
// for the next pass. It returns the deleted rows, their charged bytes and the
// rows read: no matching row is left when the first equals the last.
const boundedDeleteSQL = `
WITH candidate AS (%[2]s
), measured AS (
    SELECT row_id, charged_bytes,
        position * $4::bigint AS running_rows,
        sum(charged_bytes) OVER (ORDER BY position ROWS UNBOUNDED PRECEDING) AS running_bytes
    FROM candidate
), removed AS (
    DELETE FROM %[1]s target
    USING measured
    WHERE target.ctid = measured.row_id
        AND measured.running_rows <= $1::bigint
        AND (measured.running_bytes <= $2::bigint OR ($3::boolean AND measured.running_rows = $4::bigint))
    RETURNING measured.charged_bytes
)
SELECT count(*), COALESCE(sum(charged_bytes), 0)::bigint, (SELECT count(*) FROM candidate)
FROM removed
`

// unorderedCandidateSQL reads the first $1 + 1 rows of table %[1]s that
// match %[3]s in whatever order the plan finds them, charges each %[2]s and
// numbers them in physical order. It is the candidate statement of a bounded
// delete without an order.
const unorderedCandidateSQL = `
    SELECT row_id, charged_bytes, row_number() OVER (ORDER BY row_id) AS position
    FROM (SELECT ctid AS row_id, (%[2]s)::bigint AS charged_bytes
        FROM %[1]s
        WHERE %[3]s
        LIMIT $1::bigint + 1) unordered`

// orderedCandidateSQL reads the first $1 + 1 rows of table %[1]s that match
// %[3]s in order %[4]s, charges each %[2]s and numbers them in that order. It
// is the candidate statement of a bounded delete with an order; an index on
// the filter and order columns serves it without reading further rows.
const orderedCandidateSQL = `
    SELECT ctid AS row_id, (%[2]s)::bigint AS charged_bytes, row_number() OVER (ORDER BY %[4]s) AS position
    FROM %[1]s
    WHERE %[3]s
    ORDER BY %[4]s
    LIMIT $1::bigint + 1`
