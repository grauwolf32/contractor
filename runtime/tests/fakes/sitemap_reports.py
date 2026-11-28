"""Exact scanner report reads over the ordinary Artifact transport."""

from functools import partial
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.sitemap import SitemapClient
from contractor_runtime.sitemap_collection import SitemapCollector
from contractor_runtime.toolsets.common.report_observations import REPORT_FORMATS, prepare_report
from fakes.artifact_responses import ArtifactHTTPResponse
from fakes.sitemap import SitemapTransport


def report_bytes(kind):
    return (
        Path(__file__).parents[3] / "api/testdata/sitemap-reports" / f"{kind}-report.json"
    ).read_bytes()


class ReportTransport(SitemapTransport):
    def __init__(self):
        super().__init__()
        self.reads = []
        self.reports = {
            name: report_bytes(kind)
            for name, kind in {
                "report": "ffuf",
                "bad": "ffuf",
                "crawl": "katana",
                "good": "katana",
            }.items()
        }
        self.report_errors = {}
        self.media_type = "application/json"

    async def request(self, method, path, *, headers, body, max_response_bytes):
        if method != "GET":
            return await super().request(
                method, path, headers=headers, body=body, max_response_bytes=max_response_bytes
            )
        self.reads.append((path, max_response_bytes))
        parsed = urlsplit(path)
        assert parsed.path.startswith("/allocations/allocation/artifacts/research/")
        assert parse_qs(parsed.query) == {"revision": ["exact"]}
        name = parsed.path.rsplit("/", 1)[-1]
        if name in self.report_errors:
            return self.error(self.report_errors[name])
        if name not in self.reports:
            return self.error((404, "artifact_not_found", False))
        return ArtifactHTTPResponse(
            200,
            {
                "content-type": self.media_type,
                "etag": '"exact"',
                "x-contractor-binding-created-at": "2026-10-01T12:00:00Z",
                "x-contractor-revision-created-at": "2026-10-02T12:00:00Z",
            },
            self.reports[name],
        )


def report_collector(transport, **kwargs):
    client = ArtifactClient("allocation", transport)
    return SitemapCollector(
        SitemapClient(client),
        report_projector=partial(prepare_report, client),
        report_formats=REPORT_FORMATS,
        **kwargs,
    )
