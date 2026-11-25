"""A standalone test process owns one native engine and one transport session."""

from contractor_runtime.adapters.http_transport import HTTPTransport as AllocationTransport


class HTTPTransport(AllocationTransport):
    def __init__(self, *args, **kwargs):
        from contractor_runtime.adapters.bridge.mitmproxy import MitmproxyEngine

        self.runtime_engine = MitmproxyEngine()
        super().__init__(*args, native_engine=self.runtime_engine, **kwargs)

    async def close(self):
        await super().close()
        await self.runtime_engine.close()
