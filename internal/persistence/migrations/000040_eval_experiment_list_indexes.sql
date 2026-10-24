-- Drop the experiment list's state, dataset and control-mode page indexes,
-- whose filters the list no longer takes (V414-026, EVM-R247).
--
-- The experiment list pages every Project of an owner, or one Project, newest
-- first through eval_experiments_owner_created_page_idx and
-- eval_experiments_project_created_page_idx. Its state, datasetId and
-- controlMode filters were the only readers of the three indexes below:
-- EXPLAIN on 3 owners x 3,000 experiments, in the custom and the generic
-- plan, shows that no other production statement and no foreign key probe
-- uses them. The Eval claim reads the active and paused states through the
-- partial eval_experiments_claim_active_idx and
-- eval_experiments_claim_paused_deadline_idx; the Project fence, the Project
-- purge and blocking checks and the (owner_id, project_id) foreign key probe
-- read Project indexes or the table; the dataset revision foreign key probe
-- reads eval_experiments_dataset_revision_idx. A filter on one of these
-- columns that returns needs its own created-order index, (owner_id, column,
-- created_at DESC, experiment_id DESC), so that its page stays bounded in a
-- generic plan.
DROP INDEX eval_experiments_state_page_idx;
DROP INDEX eval_experiments_dataset_page_idx;
DROP INDEX eval_experiments_control_page_idx;
