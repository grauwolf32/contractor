package evalstore

// SQL statements for experiments.go.

// insertMembersSQL inserts one bounded batch of experiment $1's expected
// matrix (freezeBatches), so the statement trigger eval_members_projection
// runs once per batch with the batch in its transition table and advances
// the projection queue once, while the row trigger eval_members_bound still
// checks each row. $2-$16 hold one element per member in manifest order:
// member ID, pair ID, ordinal, suite ID, case ID, sample, variant ID,
// eligibility, execution kind, submission key, case and binding digests,
// eligibility reason (NULL when absent), and the 1-based positions of the
// member's case part in $17 and variant part in $18. A recipe is exactly the
// case part followed by the variant part (see memberRows), so each distinct
// projected case and each variant of the batch is sent once instead of once
// per member. Used by Store.persistPlan.
var insertMembersSQL = `
INSERT INTO eval_members(
    experiment_id, member_id, pair_id, ordinal, suite_id, case_id, sample,
    variant_id, eligibility, execution_kind, recipe, submission_key,
    case_sha256, binding_sha256, eligibility_reason
)
SELECT $1, m.member_id, m.pair_id, m.ordinal, m.suite_id, m.case_id, m.sample,
       m.variant_id, m.eligibility, m.execution_kind,
       ($17::bytea[])[m.case_part] || ($18::bytea[])[m.variant_part],
       m.submission_key, m.case_sha256, m.binding_sha256, m.eligibility_reason
  FROM unnest(
        $2::text[], $3::text[], $4::integer[], $5::text[], $6::text[],
        $7::integer[], $8::text[], $9::text[], $10::text[], $11::text[],
        $12::text[], $13::text[], $14::text[], $15::integer[], $16::integer[]
       ) WITH ORDINALITY AS m(
        member_id, pair_id, ordinal, suite_id, case_id,
        sample, variant_id, eligibility, execution_kind, submission_key,
        case_sha256, binding_sha256, eligibility_reason, case_part, variant_part,
        position
       )
 ORDER BY m.position`
