import asyncio
import websockets

async def echo_server(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(f"Echo: {message}")  # Send back the message

start_server = websockets.serve(echo_server, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
print("Server started at ws://localhost:8765")
asyncio.get_event_loop().run_forever()
