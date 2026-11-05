"""The exact disk-heavy cases whose default skips the test gates may accept."""

import os

MODULE = "github.com/grauwolf32/contractor/"
FLAG = "CONTRACTOR_TEST_HEAVY_IO"
CASES = {
    (
        "internal/auditstore/internal/pgstore",
        "TestSharedPostgresCommonTaskPagesAndPurgeAtCeiling",
    ): "100000 tasks",
    (
        "internal/auditstore/internal/pgstore",
        "TestPostgresAuditPurgeAtScaleProbesForeignKeysThroughIndexes",
    ): "150000 executions",
    (
        "internal/auditstore/internal/pgstore",
        "TestPostgresAuditReadsAtFindingScale",
    ): "60000 executions plus findings",
    (
        "internal/auditstore/internal/pgstore",
        "TestPostgresAuditClaimReadsOnlyTheReconcilableAudits",
    ): "more than 64000 Audits and claims",
    (
        "internal/artifacts/postgres",
        "TestPostgresArtifactSelectiveQueryPlans",
    ): "50000 revisions and associated blobs",
    (
        "internal/artifacts/postgres",
        "TestPostgresIntegration64MiBPayloadBoundary",
    ): "real 64-MiB payload",
    (
        "internal/artifacts/postgres",
        "TestPostgresRunRevisionLimitAndBoundedPurge",
    ): "50000 revisions",
    (
        "internal/artifacts/postgres",
        "TestPostgresRunRevisionLimitIsNotRefundedByHistoryRetention",
    ): "50000 revisions",
    (
        "internal/projectsitemap",
        "TestPostgresSitemapSingleRunExceedsRemovedPublicationCaps",
    ): "more than 256 MiB of publications",
    (
        "internal/findingintake",
        "TestPostgresHydratesNineMaximumSizeProposalsAcrossBatches",
    ): "nine maximum-size proposals",
}


def enabled() -> bool:
    value = os.environ.get(FLAG, "")
    if value not in ("", "0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    return value == "1"


def optional(package: str, case: str) -> bool:
    return (
        not enabled() and (package.removeprefix(MODULE), case.split("/", 1)[0]) in CASES
    )
