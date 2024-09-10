import asyncio
from contextlib import suppress


class gfa_loop_test:
    async def start_loop(self):
        self._check_task = asyncio.create_task(self._check_loop())
        
        return self
            
    async def stop_loop(self):
        await self.cancel_task(self._check_task)

    async def _check_loop(self):
        while True:
            print("Loop is running...")
            await asyncio.sleep(1)  # Asynchronous wait to prevent blocking

    async def cancel_task(self, task: asyncio.Future):
        """Safely cancels a task."""

        if task is None or task.done():
            return

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task