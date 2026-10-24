package runtimeconfig

// lockBindingsSQL reads the bindings of one label set, $1 sorted bytewise
// and unique, in that order. The caller appends the lock mode (FOR SHARE or
// FOR UPDATE) and OF binding: PostgreSQL then takes each row lock as the
// sorted rows leave, so the order is the label array's order rather than the
// database collation's, and a missing label simply returns no row.
// Used by Repository.lockBindings.
const lockBindingsSQL = `
SELECT binding.label, binding.config_name, binding.config_version, binding.config_digest,
       binding.revision::text, binding.created_by, binding.created_at,
       binding.updated_by, binding.updated_at
FROM runtime_label_bindings AS binding
JOIN unnest($1::text[]) WITH ORDINALITY AS requested(label, position)
  ON requested.label = binding.label
ORDER BY requested.position
`
