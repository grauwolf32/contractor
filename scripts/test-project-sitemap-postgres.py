#!/usr/bin/env python3
"""Run complete sitemap owner packages and require every real-store case."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import disk_heavy_policy

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "github.com/grauwolf32/contractor/"
PACKAGES = (
    "internal/scheduler", "internal/operations", "internal/operations/sitemapjoin",
    "internal/projectsitemap", "internal/artifacts/postgres", "internal/projectstore",
    "internal/projectlifecycle", "internal/persistence/postgres",
    "internal/httpapi/public", "internal/cli", "internal/publicclient",
    "internal/app", "internal/artifacts/policy", "internal/archtest", "internal/httpapi/privateartifacts", "internal/controlplane", "internal/runstore",
)
REQUIRED = {
    "TestPostgresSitemapReceiptsOwnershipAndHistory",
    "TestPostgresSitemapHTTPCollectionProjectionAndReceipt",
    "TestPostgresSitemapRepeatedRunsAggregateBehavior",
    "TestPostgresSitemapPreparedBatchReplaysAfterEvidenceAndMapEviction",
    "TestPostgresSitemapObservationsAggregateAcrossRunsAndBoundEvidence",
    "TestPostgresSitemapObservationBatchBoundaryAndRollback",
    "TestPostgresSitemapObservationEvidenceVisibilityAndProjectLifecycle",
    "TestPostgresSitemapSnapshotRebindsReportEvidenceToAuthorizedOriginalInput",
    "TestPostgresSitemapNormalizedObservationBatchRejectsInvalidRows",
    "TestPostgresSitemapManualRunsShareMetadataAndIsolateReceipts",
    "TestPostgresSitemapSharedMapSnapshotAndReplay",
    "TestPostgresSitemapEditorsShareMembersAndDeriveProjectAuthority",
    "TestPostgresSitemapReadsDoNotResolveSnapshotInputs",
    "TestPostgresSitemapSnapshotMissingContentReportsStorageFailure",
    "TestPostgresSitemapContextProjectionSerializesWithHistoryEviction",
    "TestPostgresSitemapSnapshotMaterializesOnlyNeededAuthorizedExactInputs",
    "TestPostgresSitemapTenWritersAndPermanentSourceConflict",
    "TestPostgresSitemapMalformedAndProjectFence",
    "TestPostgresSitemapSingleRunExceedsRemovedPublicationCaps",
    "TestPostgresSitemapSelectedContextUsesOrdinaryExactForkHold",
    "TestPostgresSitemapReplayKeepsOriginalStructuralPolicy",
    "TestPostgresSitemapUsesSharedTransferAdmission",
    "TestPostgresSitemapCommitsCommonMaintenanceAfterRollback",
}

REQUIRED_CASES = {(PREFIX + "internal/scheduler", case) for case in {
    "TestPostgresServerOperationStagePublishesWithoutPlannerOrRuntime",
    "TestPostgresServerOperationRecoveryReplaysReceipt",
    "TestPostgresServerOperationMaterializesOnlyDeclaredOutputs",
    "TestPostgresServerOperationClaimFenceRejectsOldOwner",
    "TestPostgresServerOperationPreservesAppendsAndRejectsChangedIntent",
}} | {(PREFIX + "internal/projectsitemap", case) for case in REQUIRED} | {
    (PREFIX + "internal/httpapi/privateartifacts", "TestPostgresPrivateSitemapRolesSafeBytesScopeAndFence"),
    (PREFIX + "internal/httpapi/privateartifacts", "TestPostgresPrivateSitemapImportAuthorityAndExactSources"),
    (PREFIX + "internal/controlplane", "TestSitemapGrantPinsSelectedOperationsAndNeverAliasesPolicy"),
    (PREFIX + "internal/httpapi/privateartifacts", "TestPostgresPrivateProjectArtifactReadsUseConcreteTargetAndNoCacheHolds"),
    (PREFIX + "internal/runstore", "TestPostgresRunProjectContextUsesCallerTransaction"),
    (PREFIX + "internal/artifacts/postgres", "TestPostgresDescribeForkRevisionsPreservesTargetsAndBoundsWork"),
    (PREFIX + "internal/httpapi/public", "TestPostgresPublicArtifactUploadAvailability"),
    (PREFIX + "internal/httpapi/public", "TestPostgresRepeatArtifactUploadAvailabilityFollowsSourceProject"),
    (PREFIX + "internal/runstore", "TestPostgresDescribeOriginalInputsKeepsCreationRevisionsInCallerTransaction"),
    (PREFIX + "internal/persistence/postgres", "TestPostgresProjectReceiptScopesAndStrictShape"),
    (PREFIX + "internal/persistence/postgres", "TestPostgresProjectReceiptPurgeIsBoundedAndIsolated"),
}


def main():
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise SystemExit("CONTRACTOR_TEST_DATABASE_URL is required")
    with tempfile.TemporaryDirectory(prefix="project-sitemap-postgres-") as directory:
        report = Path(directory) / "tests.jsonl"
        with report.open("w") as output:
            result = subprocess.run(
                ["go", "test", "-json", "-tags=integration", "-p=2", "-count=1",
                 "-timeout=15m", *("./" + package for package in PACKAGES)],
                cwd=ROOT, stdout=output, check=False, timeout=1200,
            )
        passed, started, packages, failures, skipped = set(), set(), set(), [], set()
        for line in report.read_text().splitlines():
            event = json.loads(line)
            package, case, action = event.get("Package"), event.get("Test"), event.get("Action")
            if action == "pass" and not case:
                packages.add(package)
            if (package, case) in REQUIRED_CASES:
                if action == "run":
                    started.add((package, case))
                if action == "pass":
                    passed.add((package, case))
                if action == "skip" and disk_heavy_policy.optional(package, case):
                    skipped.add((package, case))
                elif action in {"skip", "fail"}:
                    failures.append(f"{case}: {action}")
            if action == "output" and ("encoded_publication_bytes=" in event.get("Output", "")):
                print(event["Output"].strip(), flush=True)
        missing = REQUIRED_CASES - (started & (passed | skipped))
        missing_packages = {PREFIX + package for package in PACKAGES} - packages
        if result.returncode or missing or missing_packages or failures:
            # Connection strings are never diagnostics.
            print(report.read_text().replace(os.environ["CONTRACTOR_TEST_DATABASE_URL"], "[database]"))
            raise SystemExit(f"sitemap gate failed: {sorted(missing)}, {sorted(missing_packages)}, {failures}")
        print(f"Sitemap PostgreSQL: {len(PACKAGES)} complete packages, {len(REQUIRED_CASES) - len(skipped)} mandatory cases passed, {len(skipped)} optional disk-heavy skips")


if __name__ == "__main__":
    main()
