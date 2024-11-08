from typing import Any, Callable, Optional, Tuple, Union
from google.cloud import aiplatform
from vertexai.generative_models import (
    ChatSession,
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Part,
    Tool,
)

import os
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv('REGION')
PROJECT=os.getenv('GOOGLE_CLOUD_PROJECT')
print(PROJECT)
print(aiplatform.__version__)

class ChatAgent:
    def __init__(
        self,
        model: GenerativeModel,
        tool_handler_fn: Callable[[str, dict], Any],
        max_iterative_calls: int = 5,
    ):
        self.tool_handler_fn = tool_handler_fn
        self.chat_session = model.start_chat()
        self.max_iterative_calls = 5

    def send_message(self, message: str) -> GenerationResponse:
        response = self.chat_session.send_message(message)

        # This is None if a function call was not triggered
        fn_call = response.candidates[0].content.parts[0].function_call
        print(fn_call)

        num_calls = 0
        # Reasoning loop. If fn_call is None then we never enter this
        # and simply return the response
        while fn_call:
            if num_calls > self.max_iterative_calls:
                break

            # Handle the function call
            fn_call_response = self.tool_handler_fn(
                fn_call.name, dict(fn_call.args)
            )
            num_calls += 1

            # Send the function call result back to the model
            response = self.chat_session.send_message(
                Part.from_function_response(
                    name=fn_call.name,
                    response={
                        "content": fn_call_response,
                    },
                ),
            )

            # If the response is another function call then we want to
            # stay in the reasoning loop and keep calling functions.
            fn_call = response.candidates[0].content.parts[0].function_call

        return response

current_weather_func = FunctionDeclaration(
    name="current_weather",
    description="Get the current weather at a specified location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location",
            }
        },
        "required": ["location"],
    },
)

# Simulate a function that calls a weather API


def current_weather(location: str) -> dict:
    print("Executing current_weather function...")
    api_response = {
        "location": "New York City",
        "temperature": "55 degrees (F)",
        "wind": "8 mph",
        "wind_direction": "West",
        "skies": "clear/sunny",
        "chance_of_rain": "0%",
    }
    return api_response

current_congestion_levels_func = FunctionDeclaration(
    name="current_congestion_levels",
    description="Get the current congestion levels for buses, trains, and roads at a specified location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location",
            }
        },
        "required": ["location"],
    },
)


def current_congestion_levels(location: str) -> dict:
    print('Executing current_congestion_levels function...')
    api_response = {
        "location": "New York City",
        "bus": "Congested",
        "train": "Uncongested",
        "roads": "Congested"
    }
    return api_response
    

# Tools can wrap around one or multiple functions
confortable_travel_tool = Tool(
    function_declarations=[current_weather_func,
                           current_congestion_levels_func])


# Instantiate model with weather tool
model = GenerativeModel(
    "gemini-1.0-pro-001",
    tools=[confortable_travel_tool],
)

def weather_tool_handler_fn(fn_name: str, fn_args: dict) -> dict:
    if fn_name == "current_weather":
        return current_weather(fn_args["location"])
    elif fn_name == 'current_congestion_levels':
        return current_congestion_levels(fn_args['location'])
    else:
        raise ValueError(f"Unknown function call: {fn_name}")


chat = ChatAgent(model=model, tool_handler_fn=weather_tool_handler_fn)
response = chat.send_message("What is the weather like in New York City")
print(response.candidates[0].content.parts[0].text)