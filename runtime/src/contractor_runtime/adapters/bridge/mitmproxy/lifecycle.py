"""Join native callbacks and writers before releasing allocation resources."""

import asyncio

from contractor_runtime.core.async_tasks import join_owned_task


class OwnedHandler:
    def exchange_for_hook(self, hook):
        data = hook.args()[0]
        flows = getattr(getattr(self, "engine", None), "flows", {})
        return flows.get(getattr(data, "id", None))

    def hook_task(self, hook):
        # Called synchronously when the native dispatcher creates its callback.
        # Register now, including callbacks that have not begun executing yet.
        task = asyncio.create_task(super().hook_task(hook), name="http-engine-callback")
        self.hooks.add(task)
        exchange = self.exchange_for_hook(hook)
        if exchange is not None:
            exchange.hooks.add(task)

        async def join():
            try:
                cancelled = await join_owned_task(task)
                task.result()
                if cancelled:
                    raise asyncio.CancelledError
            finally:
                self.hooks.discard(task)

        return join()

    async def handle_connection(self, connection):
        writer = self.transports[connection].writer
        try:
            await super().handle_connection(connection)
        finally:
            # Keep the exact writer until confirmed closed; a failed wait remains
            # owned and is retried by close rather than losing the socket handle.
            self.writers.add(writer)
            await self.close_writer(writer)

    async def close_writer(self, writer):
        async def close():
            writer.close()
            await writer.wait_closed()

        task = asyncio.create_task(close(), name="http-engine-writer-close")
        cancelled = await join_owned_task(task)
        task.result()
        self.writers.discard(writer)
        if cancelled:
            raise asyncio.CancelledError

    async def join(self):
        tasks = {io.handler for io in self.transports.values() if io.handler is not None}
        tasks.update(self.hooks)
        tasks.update(self.wakeup_timer)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Even a native reader/dispatcher failure must not discard its writer.
        self.writers.update(io.writer for io in self.transports.values() if io.writer is not None)
        for writer in tuple(self.writers):
            await self.close_writer(writer)
        self.transports.clear()
        self.hooks.clear()
        self.wakeup_timer.clear()
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, OSError):
                raise result
