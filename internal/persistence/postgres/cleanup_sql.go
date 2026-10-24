package postgres

// SQL statements for cleanup.go.

// boundedDeleteSQL deletes the longest run of one table's matching rows that
// fits the rest of a cleanup pass's budget (DeleteWithinBudget). %[1]s is the
// table, %[2]s the stored size of one row and %[3]s the filter, which binds
// $5 onwards. $1 is the row mutations the pass may still make, $2 the stored
// bytes it may still remove, $3 whether it has spent nothing yet and $4 the
// row mutations one deleted row costs. At most $1 + 1 matching rows are read
// and measured before any is deleted; they are taken in physical order, a row
// joining while the running row cost stays within $1 and the running stored
// size within $2. The first row of a fresh pass joins whatever its size. A
// row a concurrent writer replaced since the read is left for the next pass.
// It returns the deleted rows, their stored bytes and the rows read: no
// matching row is left when the first equals the last.
const boundedDeleteSQL = `
WITH candidate AS (
    SELECT ctid AS row_id, (%[2]s)::bigint AS stored_bytes
    FROM %[1]s
    WHERE %[3]s
    LIMIT $1::bigint + 1
), measured AS (
    SELECT row_id, stored_bytes,
        row_number() OVER w * $4::bigint AS running_rows,
        sum(stored_bytes) OVER w AS running_bytes
    FROM candidate
    WINDOW w AS (ORDER BY row_id ROWS UNBOUNDED PRECEDING)
), removed AS (
    DELETE FROM %[1]s target
    USING measured
    WHERE target.ctid = measured.row_id
        AND measured.running_rows <= $1::bigint
        AND (measured.running_bytes <= $2::bigint OR ($3::boolean AND measured.running_rows = $4::bigint))
    RETURNING measured.stored_bytes
)
SELECT count(*), COALESCE(sum(stored_bytes), 0)::bigint, (SELECT count(*) FROM candidate)
FROM removed
`
