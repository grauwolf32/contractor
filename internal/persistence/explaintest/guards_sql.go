package explaintest

// SQL statements for guards.go.

// beforeUpdateTriggersSQL lists each BEFORE UPDATE row trigger of table $1
// with the source of its function followed by its definition, which holds
// its WHEN condition, and its UPDATE OF columns, if any.
// Used by RequireColumnScopedGuards.
var beforeUpdateTriggersSQL = `
SELECT trigger.tgname::text, function.prosrc || E'\n' || pg_get_triggerdef(trigger.oid),
       ARRAY(SELECT attribute.attname::text
               FROM unnest(trigger.tgattr::int2[]) AS column_number
               JOIN pg_attribute AS attribute
                 ON attribute.attrelid = trigger.tgrelid AND attribute.attnum = column_number)
  FROM pg_trigger AS trigger
  JOIN pg_proc AS function ON function.oid = trigger.tgfoid
 WHERE trigger.tgrelid = $1::text::regclass AND NOT trigger.tgisinternal
   -- ROW (1), BEFORE (2) and UPDATE (16) of pg_trigger.tgtype
   AND trigger.tgtype & 19 = 19
 ORDER BY 1`
