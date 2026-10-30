-- Seed-era pins describe a different execution contract. Never rewrite them.
DO $$
BEGIN
 IF EXISTS (SELECT 1 FROM audits
   WHERE baseline_snapshot #> '{engagement,document,target,seedUrls}' IS NOT NULL) THEN
  RAISE EXCEPTION 'Seed-based Audit pins require explicit operator resolution before the interactive HTTP upgrade' USING ERRCODE='23514';
 END IF;
END;
$$;

-- Historical action attribution and unresolved holds remain immutable. New
-- actions use a per-allocation request ordinal bounded by actionIntents (100000).
ALTER TABLE audit_actions DROP CONSTRAINT audit_actions_phase_check;
ALTER TABLE audit_actions ADD CONSTRAINT audit_actions_phase_check
 CHECK (phase IN ('request','seed','login','control','auth-check','refresh','refresh-control'));
ALTER TABLE audit_actions DROP CONSTRAINT audit_actions_generation_check;
ALTER TABLE audit_actions ADD CONSTRAINT audit_actions_generation_check CHECK (
 generation BETWEEN 0 AND 1 AND
 (phase NOT IN ('request','login','control') OR generation=0) AND
 (phase NOT IN ('refresh','refresh-control') OR generation=1) AND
 (identity_id<>'anonymous' OR (phase IN ('request','seed') AND generation=0)));
ALTER TABLE audit_actions DROP CONSTRAINT audit_actions_step_index_check;
ALTER TABLE audit_actions ADD CONSTRAINT audit_actions_step_index_check CHECK (
 (phase='request' AND step_index BETWEEN 0 AND 99999) OR
 (phase IN ('seed','auth-check') AND step_index BETWEEN 0 AND 127) OR
 (phase IN ('login','refresh') AND step_index BETWEEN 0 AND 15) OR
 (phase IN ('control','refresh-control') AND step_index=0));
ALTER TABLE audit_action_attempts DROP CONSTRAINT audit_action_attempts_method_check;
ALTER TABLE audit_action_attempts ADD CONSTRAINT audit_action_attempts_method_check
 CHECK (method IN ('GET','HEAD','OPTIONS','POST','PUT','PATCH','DELETE'));
ALTER TABLE audit_action_attempts DROP CONSTRAINT audit_action_attempts_action_class_check;
ALTER TABLE audit_action_attempts ADD CONSTRAINT audit_action_attempts_action_class_check CHECK (
 (method IN ('GET','HEAD','OPTIONS') AND action_class='read') OR
 (method='POST' AND action_class='probe') OR
 (method IN ('PUT','PATCH','DELETE') AND action_class='test-data-write'));
