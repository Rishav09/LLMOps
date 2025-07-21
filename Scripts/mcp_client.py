import asyncio
from fastmcp import Client

async def main():
    async with Client("HW4.py") as cli:
        print("Connecting…")
        tools = await cli.list_tools()
        print("Tools:", tools)


if __name__ == "__main__":
    asyncio.run(main())
