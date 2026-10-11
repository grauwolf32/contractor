package artifacts

// SQL statements for ref_page.go.

// listRefPageSQL reads up to $8 current bindings of scope ($1, $2) after the
// keyset position ($3, $4) in namespace and name order; the empty position
// precedes every binding. Namespaces in $5 are excluded, and outside the
// Namespaces in $6 so are names that start with any prefix in $7. The
// exclusions run before the LIMIT, so they never shorten a page. Used by
// PostgresRepository.ListRefs.
var listRefPageSQL = `
SELECT namespace, name
FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2
  AND (namespace, name) > ($3::text, $4::text)
  AND namespace <> ALL ($5::text[])
  AND (namespace = ANY ($6::text[])
       OR NOT EXISTS (SELECT 1 FROM unnest($7::text[]) AS hidden(prefix)
                      WHERE starts_with(name, hidden.prefix)))
ORDER BY namespace, name
LIMIT $8`

// listNamespaceRefPageSQL is listRefPageSQL for the single Namespace $3, after
// name $4 (an empty name precedes every name). Used by
// PostgresRepository.ListRefs.
var listNamespaceRefPageSQL = `
SELECT namespace, name
FROM artifact_bindings
WHERE scope_kind = $1 AND scope_id = $2
  AND namespace = $3 AND name > $4::text
  AND namespace <> ALL ($5::text[])
  AND (namespace = ANY ($6::text[])
       OR NOT EXISTS (SELECT 1 FROM unnest($7::text[]) AS hidden(prefix)
                      WHERE starts_with(name, hidden.prefix)))
ORDER BY name
LIMIT $8`
