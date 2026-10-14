-- A summary continuation uses the existing Stage attempt and transition
-- receipts. Retrying that same continuation preserves its strategy ordinal;
-- only the first target of the source summary consumes the escalation slot.
ALTER TABLE stage_executions
  DROP CONSTRAINT stage_execution_config_variant_shape;

ALTER TABLE stage_executions
  ADD CONSTRAINT stage_execution_config_variant_shape CHECK (
    (execution_config_variant = 'base' AND escalation_ordinal IS NULL
      AND NOT (stage_context_snapshot ? 'continuation'))
    OR (
      execution_config_variant IN ('failed_escalation', 'interrupted_escalation')
      AND escalation_ordinal IS NOT NULL AND escalation_ordinal > 0
      AND NOT (stage_context_snapshot ? 'continuation')
    )
    OR (
      execution_config_variant = 'summary_escalation'
      AND escalation_ordinal IS NOT NULL
      AND escalation_ordinal = 1
      AND stage_context_snapshot ? 'continuation'
      AND jsonb_typeof(stage_context_snapshot -> 'continuation') = 'object'
      AND (stage_context_snapshot -> 'continuation') ? 'sourceStageExecutionId'
      AND jsonb_typeof(stage_context_snapshot -> 'continuation' -> 'sourceStageExecutionId') = 'string'
    )
  );

DROP INDEX stage_execution_escalation_ordinal_unique;

CREATE UNIQUE INDEX stage_execution_escalation_ordinal_unique
  ON stage_executions (run_id, stage_name, execution_config_variant, escalation_ordinal)
  WHERE execution_config_variant <> 'base'
    AND resume_source_execution_id IS NULL
    AND (
      execution_config_variant <> 'summary_escalation'
      OR stage_context_snapshot -> 'continuation' ->> 'sourceStageExecutionId' = previous_execution_id
    );

-- A terminal summary is incomplete work, never a successful Stage result.
CREATE OR REPLACE FUNCTION contractor_valid_stage_result(payload jsonb) RETURNS boolean
    LANGUAGE sql IMMUTABLE
    AS $$
    SELECT COALESCE(
        jsonb_typeof(payload) = 'object'
        AND jsonb_typeof(payload->'apiVersion') = 'string'
        AND btrim(payload->>'apiVersion') <> ''
        AND payload->>'outcome' IN ('succeeded', 'failed')
        AND jsonb_typeof(payload->'summary') = 'string'
        AND btrim(payload->>'summary') <> ''
        AND jsonb_typeof(payload->'artifacts') = 'object'
        AND (NOT (payload ? 'summarized') OR jsonb_typeof(payload->'summarized') = 'boolean')
        AND (
            (COALESCE(payload->'summarized' = 'true'::jsonb, false)
                AND payload->>'outcome' = 'failed'
                AND payload->'error'->>'code' = 'worker_summarized'
                AND payload->'error'->'retryable' = 'false'::jsonb)
            OR
            (NOT COALESCE(payload->'summarized' = 'true'::jsonb, false)
                AND payload->'error'->>'code' IS DISTINCT FROM 'worker_summarized')
        )
        AND (
            NOT (payload ? 'summaryArtifact')
            OR (
                payload->'summarized' = 'true'::jsonb
                AND jsonb_typeof(payload->'summaryArtifact') = 'object'
                AND jsonb_typeof(payload->'summaryArtifact'->'namespace') = 'string'
                AND btrim(payload->'summaryArtifact'->>'namespace') <> ''
                AND jsonb_typeof(payload->'summaryArtifact'->'name') = 'string'
                AND btrim(payload->'summaryArtifact'->>'name') <> ''
                AND jsonb_typeof(payload->'summaryArtifact'->'revision') = 'string'
                AND btrim(payload->'summaryArtifact'->>'revision') <> ''
            )
        )
        AND (
            (payload->>'outcome' = 'succeeded'
                AND (NOT (payload ? 'error') OR payload->'error' = 'null'::jsonb))
            OR
            (payload->>'outcome' = 'failed'
                AND jsonb_typeof(payload->'error') = 'object'
                AND jsonb_typeof(payload->'error'->'code') = 'string'
                AND btrim(payload->'error'->>'code') <> ''
                AND jsonb_typeof(payload->'error'->'message') = 'string'
                AND btrim(payload->'error'->>'message') <> ''
                AND jsonb_typeof(payload->'error'->'retryable') = 'boolean')
        ),
        false
    )
$$;

