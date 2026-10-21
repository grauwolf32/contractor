package settingsstore

// updateModelSelectionSQL performs a revision-guarded replacement. Equal
// values at the same revision return the unchanged row without firing the
// semantic-update trigger; a stale revision returns no row.
var updateModelSelectionSQL = `
WITH changed AS (
    UPDATE model_selection_settings
       SET default_model = $1,
           role_models = $2::jsonb,
           revision = revision + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE singleton = true AND revision = $3::numeric
       AND (default_model IS DISTINCT FROM $1 OR role_models IS DISTINCT FROM $2::jsonb)
    RETURNING default_model, role_models, revision::text, updated_at
), unchanged AS (
    SELECT default_model, role_models, revision::text, updated_at
      FROM model_selection_settings
     WHERE singleton = true AND revision = $3::numeric
       AND default_model IS NOT DISTINCT FROM $1 AND role_models IS NOT DISTINCT FROM $2::jsonb
)
SELECT default_model, role_models, revision, updated_at FROM changed
UNION ALL
SELECT default_model, role_models, revision, updated_at FROM unchanged
LIMIT 1`
