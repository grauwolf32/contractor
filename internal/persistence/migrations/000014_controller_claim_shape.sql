-- One shape for every controller claim table (V370-007, PER-R25).
--
-- The Audit and the Eval controller lease their subjects through one helper,
-- persistencepostgres.ControllerClaims, which reads and writes the same
-- columns in both claim tables. eval_controller_claims lacked the lease start
-- and the holder bound of audit_controller_claims; it gets both, and the same
-- shape rule: a free claim has no holder, start or end, a held one has all
-- three with its end after its start.
--
-- A lease does not survive this migration: held Eval claims are released with
-- their epoch kept, so the controller leases them again at a higher epoch and
-- a write under an old lease finds no live claim.

UPDATE eval_controller_claims
   SET holder_id = NULL, expires_at = NULL
 WHERE holder_id IS NOT NULL OR expires_at IS NOT NULL;

ALTER TABLE eval_controller_claims
    ADD COLUMN claimed_at timestamp with time zone,
    DROP CONSTRAINT eval_controller_claims_check,
    ADD CONSTRAINT eval_controller_claim_shape CHECK (
        (holder_id IS NULL AND claimed_at IS NULL AND expires_at IS NULL)
        OR (holder_id IS NOT NULL AND claimed_at IS NOT NULL AND expires_at > claimed_at)
    ),
    ADD CONSTRAINT eval_controller_claims_holder_id_check CHECK (
        holder_id IS NULL OR (btrim(holder_id) <> '' AND octet_length(holder_id) <= 256)
    );
