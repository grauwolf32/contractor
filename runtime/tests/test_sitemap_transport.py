"""Current sitemap authority and exact revision transport without view policies."""

import asyncio
import json

import pytest
from fakes.sitemap import SitemapTransport

from contractor_runtime.artifacts import ArtifactClient, ArtifactTransportError
from contractor_runtime.contracts import PrivateProtocolDecodeError
from contractor_runtime.sitemap import SitemapClient


def test_view_needs_only_exact_revision_metadata_and_access_needs_no_query():
    async def scenario():
        transport = SitemapTransport()
        client = SitemapClient(ArtifactClient("allocation", transport))
        view = await client.view("revision-original")
        assert view.revision == "revision-original"
        assert json.loads(view.data) == transport.document
        access = await client.access()
        assert access.policy.collect_observations is False
        assert transport.requests[-1][1] == "/allocations/allocation/sitemap/access"
        transport.revision = "another-revision"
        with pytest.raises(ArtifactTransportError, match="invalid view metadata"):
            await client.view("revision-original")

    asyncio.run(scenario())


@pytest.mark.parametrize("obsolete", [{"policyDigest": "sha256:" + "a" * 64}, {"inputs": []}])
def test_access_rejects_obsolete_projection_metadata(obsolete):
    class ObsoleteAccess(SitemapTransport):
        async def request(self, method, path, **kwargs):
            response = await super().request(method, path, **kwargs)
            return self.json({**json.loads(response.body), **obsolete})

    async def scenario():
        client = SitemapClient(ArtifactClient("allocation", ObsoleteAccess()))
        with pytest.raises(PrivateProtocolDecodeError):
            await client.access()

    asyncio.run(scenario())
