from langchain.tools import tool


@tool(description="""If the user is making a general comment about the weather (e.g., It's hot today!" or "I love rainy days"), 
                respond conversationally without asking for a location 
                or calling the weather tool.) Only ask for location or use the weather tool if the 
                user explicitly requests current weather information for a city or place.")""")
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny and 25°C."
