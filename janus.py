import asyncio
import janus

def threaded(sync_q):
    for i in range(100):
        sync_q.put(i)
    sync_q.join()

async def async_coro(async_q):
    for i in range(100):
        val = await async_q.get()
        assert val == i
        async_q.task_done()

async def main():
    queue = janus.Queue()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, threaded, queue.sync_q)
    await async_coro(queue.async_q)
    await fut
    queue.close()
    await queue.wait_closed()

asyncio.run(main())