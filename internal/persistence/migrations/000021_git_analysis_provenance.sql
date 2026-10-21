-- Keep the coverage summary on the immutable version, alongside its Git origin.
-- The complete bounded exclusion report lives in that version's source ZIP.
ALTER TABLE artifact_versions ADD COLUMN git_analysis jsonb;
ALTER TABLE artifact_versions ADD CONSTRAINT artifact_versions_git_analysis CHECK (
    git_analysis IS NULL OR (
        git_resolved_commit IS NOT NULL
        AND jsonb_typeof(git_analysis) = 'object'
        AND git_analysis ?& ARRAY['policy','reportPath','includedFiles','excludedEntries']
        AND (git_analysis - ARRAY['policy','reportPath','includedFiles','excludedEntries']) = '{}'::jsonb
        AND git_analysis->>'policy' = 'source-analysis-v1'
        AND git_analysis->>'reportPath' = '.contractor-import-report.json'
        AND jsonb_typeof(git_analysis->'includedFiles') = 'number'
        AND (git_analysis->>'includedFiles') ~ '^[1-9][0-9]{0,4}$'
        AND (git_analysis->>'includedFiles')::numeric BETWEEN 1 AND 49999
        AND jsonb_typeof(git_analysis->'excludedEntries') = 'number'
        AND (git_analysis->>'excludedEntries') ~ '^(0|[1-9][0-9]{0,4})$'
        AND (git_analysis->>'excludedEntries')::numeric BETWEEN 0 AND 50000
    ) IS TRUE
);
