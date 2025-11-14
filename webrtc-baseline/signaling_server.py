# signaling_server.py
# NOTE: This file is NOT needed when using TcpSocketSignaling!
# TcpSocketSignaling has its own built-in server/client logic.
# This file is kept for reference but should NOT be run with sender.py/receiver.py
# If you want to use this custom server, you'd need to modify sender.py/receiver.py
# to use a custom signaling implementation instead of TcpSocketSignaling.

import asyncio, json

clients = set()

async def handle(reader, writer):
    clients.add(writer)
    addr = writer.get_extra_info('peername')
    print("client connected:", addr)
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            # Relay this line to all *other* clients
            for w in list(clients):
                if w is not writer:
                    w.write(line)
                    await w.drain()
    finally:
        print("client disconnected:", addr)
        clients.discard(writer)
        writer.close()
        await writer.wait_closed()

async def main():
    server = await asyncio.start_server(handle, '127.0.0.1', 9999)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print("listening on", addrs)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
