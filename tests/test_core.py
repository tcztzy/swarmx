import json
from unittest.mock import Mock

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from swarmx import Agent, Swarm

from .conftest import create_mock_response, create_mock_streaming_response


def test_run_with_simple_message(mock_openai_client, DEFAULT_RESPONSE_CONTENT: str):
    agent = Agent()
    # set up client and run
    client = Swarm(client=mock_openai_client)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = client.run(agent=agent, model="test", messages=messages)

    # assert response content
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_tool_call(mock_openai_client, DEFAULT_RESPONSE_CONTENT: str):
    expected_location = "San Francisco"

    # set up mock to record function calls
    get_weather_mock = Mock()

    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

    agent = Agent(name="Test Agent", functions=[get_weather])
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    # set mock to return a response that triggers function call
    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    # set up client and run
    client = Swarm(client=mock_openai_client)
    response = client.run(agent=agent, model="test", messages=messages)

    get_weather_mock.assert_called_once_with(location=expected_location)
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_execute_tools_false(mock_openai_client, DEFAULT_RESPONSE_CONTENT: str):
    expected_location = "San Francisco"

    # set up mock to record function calls
    get_weather_mock = Mock()

    def get_weather(location):
        get_weather_mock(location=location)
        return "It's sunny today."

    agent = Agent(name="Test Agent", functions=[get_weather])
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]

    # set mock to return a response that triggers function call
    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": expected_location}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    # set up client and run
    client = Swarm(client=mock_openai_client)
    response = client.run(
        agent=agent, model="test", messages=messages, execute_tools=False
    )

    # assert function not called
    get_weather_mock.assert_not_called()

    # assert tool call is present in last response
    tool_calls = response.messages[-1].get("tool_calls")
    assert tool_calls is not None
    tool_calls = list(tool_calls)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "location": expected_location
    }


def test_handoff(mock_openai_client, DEFAULT_RESPONSE_CONTENT: str):
    def transfer_to_agent2():
        return agent2

    agent1 = Agent(name="Test Agent 1", functions=[transfer_to_agent2])
    agent2 = Agent(name="Test Agent 2")

    # set mock to return a response that triggers the handoff
    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "transfer_to_agent2"}],
            ),
            create_mock_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            ),
        ]
    )

    # set up client and run
    client = Swarm(client=mock_openai_client)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "I want to talk to agent 2"}
    ]
    response = client.run(agent=agent1, model="test", messages=messages)

    assert response.agent == agent2
    assert response.messages[-1]["role"] == "assistant"
    assert response.messages[-1]["content"] == DEFAULT_RESPONSE_CONTENT


def test_streaming(mock_openai_client, DEFAULT_RESPONSE_CONTENT: str):
    mock_openai_client.set_sequential_responses(
        [
            create_mock_streaming_response(
                {"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT}
            )
        ]
    )
    client = Swarm(client=mock_openai_client)
    agent = Agent()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = client.run(agent=agent, model="test", messages=messages, stream=True)
    for i, chunk in enumerate(response):
        match chunk:
            case {"delim": "start"}:
                assert i == 0
            case {"delim": "end"}:
                assert i == len(DEFAULT_RESPONSE_CONTENT.split()) + 1
            case {"response": response}:
                ...
            case _:
                assert chunk["content"] in DEFAULT_RESPONSE_CONTENT  # type: ignore
