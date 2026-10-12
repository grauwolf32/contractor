-- Index the probe holder of a model recovery route (V368-034).
--
-- Ending a Stage, discarding a placement and admitting a Run find the probes
-- a Run holds or reserved by probe_run_id (SCH-R106, SCH-R107). Recovery
-- routes are retained (SCH-R169), so without an index each of those
-- statements read every route ever created. Only a route with a probe in
-- flight or reserved has a holder, so the partial index stays as small as
-- the probes themselves.
CREATE INDEX gateway_recovery_routes_probe_run_idx
    ON gateway_recovery_routes USING btree (probe_run_id)
    WHERE probe_run_id IS NOT NULL;
