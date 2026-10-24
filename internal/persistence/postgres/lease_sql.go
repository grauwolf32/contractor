package postgres

// SQL statements for lease.go.

// leaseHeldSQL reports whether this database grants the session advisory
// lease of key $1 to the backend whose process ID is $2 and whose session
// started at $3 (PER-R34). pg_locks reads the lock table live rather than
// from the transaction's snapshot and lists the locks of every database of
// the cluster, so the statement names this database. A bigint advisory key
// appears as its high and low 32 bits in classid and objid, with objsubid 1.
// The start time tells the lease session apart from a later backend that
// reuses its process ID.
const leaseHeldSQL = `
SELECT EXISTS (
    SELECT 1
    FROM pg_locks AS lease
    JOIN pg_stat_activity AS holder ON holder.pid = lease.pid
    WHERE lease.locktype = 'advisory' AND lease.granted AND lease.objsubid = 1
      AND lease.database = (SELECT oid FROM pg_database WHERE datname = current_database())
      AND lease.classid::bigint = (($1::bigint >> 32) & 4294967295)
      AND lease.objid::bigint = ($1::bigint & 4294967295)
      AND lease.pid = $2 AND holder.backend_start = $3)`
