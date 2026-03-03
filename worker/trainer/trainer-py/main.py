import asyncio
import os

import websockets

async def handle_connection(websocket):
    """
    Handle incoming websocket messages.
    """
    print(f"Client connected: {websocket.remote_address}")

    try:
        async for message in websocket:
            await websocket.send(f"ack: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"Client disconnected: {websocket.remote_address}")


async def run_server(host: str, port: int):
    """
    Start websocket server and keep it running indefinitely.
    """
    print(f"Starting websocket server on {host}:{port}")
    async with websockets.serve(handle_connection, host, port):
        await asyncio.Future()  # Run forever.


def main():
    """
    Starts the websocket server.
    """
    host = os.getenv("WS_HOST", "0.0.0.0")
    port = int(os.getenv("WS_PORT", "8765"))
    asyncio.run(run_server(host, port))


if __name__ == "__main__":
    main()
