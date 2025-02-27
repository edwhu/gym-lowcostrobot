import asyncio
import websockets
import json

"""Run this in a separate terminal, and then open gamepad_client.html on the browser. It will then start capturing the gamepad inputs and sending them to the server. The server will then broadcast the inputs to all connected clients.
"""

connected_clients = set()

async def handle_connection(websocket, path=None):
    """
    Handle incoming WebSocket connections.
    """
    print("Browser connected.")
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            print("Received message from browser:", message)  # Log the received message
            inputs = json.loads(message)
            # Broadcast the message to all connected clients
            for client in connected_clients:
                await client.send(json.dumps(inputs))
    except websockets.ConnectionClosed:
        print("Browser disconnected.")
    finally:
        connected_clients.remove(websocket)

async def start_server():
    """
    Start the WebSocket server.
    """
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(start_server())