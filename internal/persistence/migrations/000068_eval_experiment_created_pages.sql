-- Public experiment lists page by the immutable (created_at, experiment_id)
-- key. updated_at moves on usage ticks that deliberately do not fence the
-- collection revision, so keying on it let rows move between pages.
CREATE INDEX eval_experiments_owner_created_page_idx ON eval_experiments(owner_id,created_at DESC,experiment_id DESC);
CREATE INDEX eval_experiments_project_created_page_idx ON eval_experiments(owner_id,project_id,created_at DESC,experiment_id DESC);
DROP INDEX eval_experiments_owner_updated_page_idx;
DROP INDEX eval_experiments_project_updated_page_idx;
