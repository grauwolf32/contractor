package explaintest

// SQL statements for guards.go.

// beforeUpdateTriggersSQL lists each BEFORE UPDATE row trigger of table $1
// with the language and source of its function, its definition, which holds
// its WHEN condition, and its UPDATE OF columns, if any. Used by
// RequireColumnScopedGuards.
// Trigger functions cannot have LANGUAGE sql SQL-standard bodies, so this
// trigger-only reader can use prosrc directly. Ordinary routine inspections
// use RoutineBodySQL instead, including helpers a trigger calls.
var beforeUpdateTriggersSQL = `
SELECT trigger.tgname::text, language.lanname::text, function.prosrc, pg_get_triggerdef(trigger.oid),
       ARRAY(SELECT attribute.attname::text
               FROM unnest(trigger.tgattr::int2[]) AS column_number
               JOIN pg_attribute AS attribute
                 ON attribute.attrelid = trigger.tgrelid AND attribute.attnum = column_number)
  FROM pg_trigger AS trigger
  JOIN pg_proc AS function ON function.oid = trigger.tgfoid
  JOIN pg_language AS language ON language.oid = function.prolang
 WHERE trigger.tgrelid = $1::text::regclass AND NOT trigger.tgisinternal
   -- ROW (1), BEFORE (2) and UPDATE (16) of pg_trigger.tgtype
   AND trigger.tgtype & 19 = 19
 ORDER BY 1`

// tableColumnsSQL lists the columns of table $1. Used by tableColumns.
var tableColumnsSQL = `
SELECT attname::text FROM pg_attribute
 WHERE attrelid = $1::text::regclass AND attnum > 0 AND NOT attisdropped`

// columnTypeSQL returns the type of column $2 of table $1. Used by Changed.
var columnTypeSQL = `
SELECT format_type(atttypid, atttypmod) FROM pg_attribute
 WHERE attrelid = $1::text::regclass AND attname = $2 AND attnum > 0 AND NOT attisdropped`
