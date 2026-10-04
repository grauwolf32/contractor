package auditstore

import "context"

// InvalidateDeletedRunProjections advances the revisions of Audits whose
// external Run availability changed. It deliberately does not append an
// Audit event: no finding assessment or review subject changed. A finalizing
// Audit's updated_at remains frozen for its immutable report snapshot.
func (s *PostgresStore) InvalidateDeletedRunProjections(
	ctx context.Context, auditIDs []string,
) error {
	_, err := s.db.Exec(ctx, `
UPDATE audits
   SET revision = revision + 1,
       updated_at = CASE WHEN state = 'finalizing' THEN updated_at
           ELSE GREATEST(clock_timestamp(), updated_at + interval '1 microsecond') END
 WHERE audit_id = ANY($1::text[])`, auditIDs)
	return err
}
