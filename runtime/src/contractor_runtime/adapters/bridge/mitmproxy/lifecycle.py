"""Join native callbacks and writers before releasing allocation resources."""

import asyncio

from contractor_runtime.core.async_tasks import join_owned_task


class OwnedHandler:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection_tasks = set()
        self._writer_closes = {}

    def own_connection_task(self):
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("native HTTP connection has no task")
        if task not in self._connection_tasks:
            self._connection_tasks.add(task)
            task.add_done_callback(self._connection_done)

    def _connection_done(self, task):
        # Native code removes ConnectionIO before our finally finishes. Observe
        # every outcome, retaining failed tasks until join reports their errors.
        if task.cancelled() or task.exception() is None:
            self._connection_tasks.discard(task)

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
        self.own_connection_task()
        writer = self.transports[connection].writer
        try:
            await super().handle_connection(connection)
        finally:
            # Keep the exact writer until confirmed closed; a failed wait remains
            # owned and is retried by close rather than losing the socket handle.
            self.writers.add(writer)
            await self.close_writer(writer)

    async def close_writer(self, writer):
        task = self._writer_closes.get(writer)
        if task is None or task.done():
            task = asyncio.create_task(self._close_writer(writer), name="http-engine-writer-close")
            self._writer_closes[writer] = task
        cancelled = await join_owned_task(task)
        if self._writer_closes.get(writer) is task:
            self._writer_closes.pop(writer)
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close_writer(self, writer):
        # Keep the original socket object: transports may erase extra_info at
        # connection_lost. is_closing only means shutdown started, not finished.
        sock = writer.get_extra_info("socket")
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError):
            # CPython's completed close waiter retains the network error forever.
            # Only classify it after the joined wait AND actual descriptor close.
            if sock is None or sock.fileno() != -1:
                raise
        self.writers.discard(writer)

    async def join(self):
        tasks = {io.handler for io in self.transports.values() if io.handler is not None}
        tasks.update(self._connection_tasks)
        tasks.update(self.hooks)
        tasks.update(self.wakeup_timer)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._connection_tasks.difference_update(tasks)
        # Even a native reader/dispatcher failure must not discard its writer.
        self.writers.update(io.writer for io in self.transports.values() if io.writer is not None)
        for writer in tuple(self.writers):
            await self.close_writer(writer)
        self.transports.clear()
        self.hooks.clear()
        self.wakeup_timer.clear()
        for result in results:
            if isinstance(result, Exception):
                raise result
