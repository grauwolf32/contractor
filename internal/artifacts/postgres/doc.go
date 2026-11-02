// Package postgres implements the Artifact registry on a pool or caller-owned
// transaction. It owns binding revisions, lineage, history admission and purge;
// the physical payload backend never changes their PostgreSQL authority.
//
// PreparePayload stages bytes and verifies any reusable physical generation
// before publication locks. Write publishes that candidate under the ordinary
// CAS precondition. A caller that receives HistoryMaintenanceError from its
// transaction rolls back before asking a pool repository to MaintainHistory.
// Physical deletion runs only after a definite commit; uncertain and displaced
// file generations remain for the operator's offline cleanup.
package postgres
