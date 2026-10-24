package agentprincipal

// lockPrincipalsSQL locks the principals of one set, $1 sorted bytewise and
// unique, FOR UPDATE in that order: PostgreSQL takes each row lock as the
// sorted rows leave, so the order is the ID array's order rather than the
// database collation's, and a missing principal simply returns no row.
// Used by Repository.LockMany.
const lockPrincipalsSQL = `
SELECT principal.runtime_agent_id, principal.labels, principal.label_revision::text,
       principal.created_by, principal.created_at, principal.updated_by, principal.updated_at
FROM runtime_agent_principals AS principal
JOIN unnest($1::text[]) WITH ORDINALITY AS requested(runtime_agent_id, position)
  ON requested.runtime_agent_id = principal.runtime_agent_id
ORDER BY requested.position
FOR UPDATE OF principal`
