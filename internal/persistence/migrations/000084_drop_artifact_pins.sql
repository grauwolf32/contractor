-- artifact_pins recorded which exact revisions a Run used, but nothing read
-- the rows and every purge deleted them before the revisions they named.
-- Exact refs live in stage snapshots, lineage and finding receipts; callers
-- now verify and key-share lock the revision instead of inserting a pin.
DROP TABLE artifact_pins;
