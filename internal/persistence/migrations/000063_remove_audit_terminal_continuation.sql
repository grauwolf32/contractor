-- Resume only applies to paused Audits. Terminal deadline closures remain final.
ALTER TABLE audits DROP COLUMN continuation_count;
