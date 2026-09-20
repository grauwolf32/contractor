CREATE TABLE gateway_recovery_routes (
    route_key text PRIMARY KEY,
    owner_id text NOT NULL,
    blocked boolean NOT NULL DEFAULT false,
    failure_code text NOT NULL DEFAULT '',
    failure_count bigint NOT NULL DEFAULT 0,
    blocked_at timestamptz,
    next_probe_at timestamptz,
    automatic_until timestamptz,
    probe_id text,
    probe_run_id text,
    probe_until timestamptz
);
CREATE TABLE gateway_run_routes (
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    route_key text NOT NULL REFERENCES gateway_recovery_routes(route_key),
    PRIMARY KEY (run_id,route_key)
);
CREATE TABLE gateway_allocation_routes (
    allocation_id text NOT NULL REFERENCES stage_allocations(allocation_id) ON DELETE CASCADE,
    model text NOT NULL,
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    route_key text NOT NULL REFERENCES gateway_recovery_routes(route_key),
    PRIMARY KEY(allocation_id,model)
);
CREATE TABLE gateway_recovery_waits (
    participant_id text NOT NULL,
    model text NOT NULL,
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    route_key text NOT NULL REFERENCES gateway_recovery_routes(route_key),
    PRIMARY KEY(participant_id,model)
);
CREATE INDEX gateway_recovery_waits_route_idx ON gateway_recovery_waits(route_key);

-- Safe physical-failure receipts make response-loss retries idempotent even
-- when failures from other invocations interleave. No provider content is kept.
CREATE TABLE gateway_recovery_failures (
    route_key text NOT NULL REFERENCES gateway_recovery_routes(route_key),
    request_id text NOT NULL,
    run_id text NOT NULL REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    PRIMARY KEY(route_key,request_id)
);
