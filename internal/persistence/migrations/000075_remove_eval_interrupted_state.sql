-- Managed Eval experiments cannot enter interrupted; keep the database state
-- constraint aligned with the coordinator and the public contract.
ALTER TABLE eval_experiments
    DROP CONSTRAINT eval_experiments_state_check,
    ADD CONSTRAINT eval_experiments_state_check
        CHECK (state IN ('draft','preparing','ready','running','settling','finished','pausing','paused','cancelling','cancelled'));
