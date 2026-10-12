package postgres

import (
	"context"
	"strconv"
	"strings"
)

// AdvisoryKey names one transaction-scoped advisory lock. NewAdvisoryKey is
// the only way the Server builds one (PER-R23): a namespace, which says what
// the lock serializes, and the parts that name one instance of it, such as
// an owner and an idempotency key. Every value is written after its byte
// length, so two different part lists never share a key, whatever
// characters they hold. PostgreSQL hashes the text into the lock's 64-bit
// key; a collision of two hashes only makes two unrelated holders wait for
// each other, it never lets a race through.
//
// Two locks are not keyed this way: the session-level lease of the active
// Server (PER-R2), whose fixed key every Server version shares, and the
// migration lock, which is scoped to the schema being migrated.
type AdvisoryKey string

// NewAdvisoryKey builds the advisory key of namespace and parts.
func NewAdvisoryKey(namespace string, parts ...string) AdvisoryKey {
	var builder strings.Builder
	for _, value := range append([]string{namespace}, parts...) {
		builder.WriteString(strconv.Itoa(len(value)))
		builder.WriteByte(':')
		builder.WriteString(value)
	}
	return AdvisoryKey(builder.String())
}

// AdvisoryLockSQL is the SQL call that takes the exclusive transaction
// advisory lock of the key bound to placeholder. A statement that must hold
// the lock while it writes, even on a pool where the statement is its own
// transaction, embeds it; other callers use LockAdvisory.
func AdvisoryLockSQL(placeholder string) string {
	return "pg_advisory_xact_lock(hashtextextended(" + placeholder + "::text, 0))"
}

// LockAdvisory takes key's exclusive transaction advisory lock. It lasts until
// the transaction of db ends, so db must be a transaction: on a pool the lock
// would end with this statement.
func LockAdvisory(ctx context.Context, db DBTX, key AdvisoryKey) error {
	_, err := db.Exec(ctx, `SELECT `+AdvisoryLockSQL("$1"), string(key))
	return err
}

// LockAdvisoryShared takes key's shared transaction advisory lock, which waits
// only for a holder of the exclusive lock, under the same rule as
// LockAdvisory.
func LockAdvisoryShared(ctx context.Context, db DBTX, key AdvisoryKey) error {
	_, err := db.Exec(ctx, `SELECT pg_advisory_xact_lock_shared(hashtextextended($1::text, 0))`, string(key))
	return err
}
