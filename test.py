import asyncio
import websockets

async def send_message():
    async with websockets.connect("ws://192.168.8.119:8765") as websocket:
        message = input("Enter message to send: ")
        await websocket.send(message)
        print("Message sent")

asyncio.run(send_message())
