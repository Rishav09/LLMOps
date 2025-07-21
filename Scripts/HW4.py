import random
from fastmcp import FastMCP

known_weather_data: dict[str, float] = {
    "london": 21.5,
    "mumbai": 30.2,
    "new york": 26.0,
    'berlin': 20.0,
}

mcp = FastMCP("Demo 🚀")

@mcp.tool
def get_weather(city: str) -> float:
    city = city.strip().lower()
    return known_weather_data.get(city, round(random.uniform(-5, 35), 1))

@mcp.tool
def set_weather(city: str, temp: float) -> str:
    known_weather_data[city.strip().lower()] = temp
    return "OK"



# ── If you were running this as a normal script instead ──────────────
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_async())