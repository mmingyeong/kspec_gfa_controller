import asyncio

is_loop_running = True

async def loop_function(is_loop_running):
    while is_loop_running:
        print("Loop is running...")
        await asyncio.sleep(1)  # Asynchronous wait to prevent blocking
        if not is_loop_running:
            break
        
async def stop_loop():
    is_loop_running = False
    