from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from box_box_bot.agent import graph


def test_build_agent_uses_full_history_output_mode():
    # Regression test: create_supervisor defaults to output_mode="last_message",
    # which only passes each specialist's final answer up to the supervisor -
    # not their tool calls. That silently breaks citation extraction
    # (agent/citations.py) and cost accounting (agent/cost.py), since both
    # scan the full message list for specific tool/AI messages.
    with (
        patch.object(graph, "ChatAnthropic"),
        patch.object(graph, "create_supervisor") as mock_create_supervisor,
    ):
        mock_create_supervisor.return_value = MagicMock()

        graph.build_agent()

        _, kwargs = mock_create_supervisor.call_args
        assert kwargs["output_mode"] == "full_history"


def test_build_agent_uses_a_callable_prompt_not_a_static_string():
    # Regression test: a static SUPERVISOR_PROMPT string would go stale
    # the longer the cached agent's server process stays up (it never
    # restates today's date). create_supervisor's `prompt` accepts a
    # callable, computed fresh on every model call - must not regress to
    # passing the raw string directly.
    with (
        patch.object(graph, "ChatAnthropic"),
        patch.object(graph, "create_supervisor") as mock_create_supervisor,
    ):
        mock_create_supervisor.return_value = MagicMock()

        graph.build_agent()

        _, kwargs = mock_create_supervisor.call_args
        assert callable(kwargs["prompt"])


def test_supervisor_prompt_preserves_the_users_message():
    # Regression test: create_react_agent's callable `prompt` must return
    # the FULL message list (system message + existing conversation), not
    # just the system prompt text - returning a bare string once replaced
    # the entire model input, so the user's own message never reached the
    # model at all (it just introduced itself instead of answering).
    human_message = HumanMessage(content="What are the current standings?")
    state = {"messages": [human_message]}

    result = graph._supervisor_prompt(state)

    assert isinstance(result[0], SystemMessage)
    assert "Today's date is" in result[0].content
    assert graph.SUPERVISOR_PROMPT in result[0].content
    assert result[1] is human_message


def test_build_agent_passes_all_specialists():
    with (
        patch.object(graph, "ChatAnthropic"),
        patch.object(graph, "create_supervisor") as mock_create_supervisor,
    ):
        mock_create_supervisor.return_value = MagicMock()

        graph.build_agent()

        (agents,), _ = mock_create_supervisor.call_args
        names = {agent.name for agent in agents}
        assert names == {"stats_agent", "narrative_agent", "predictor_agent"}


def test_build_agent_compiles_with_checkpointer():
    with (
        patch.object(graph, "ChatAnthropic"),
        patch.object(graph, "create_supervisor") as mock_create_supervisor,
    ):
        mock_workflow = MagicMock()
        mock_create_supervisor.return_value = mock_workflow

        graph.build_agent()

        mock_workflow.compile.assert_called_once()
        _, compile_kwargs = mock_workflow.compile.call_args
        assert isinstance(compile_kwargs["checkpointer"], InMemorySaver)


def test_build_agent_returns_compiled_workflow():
    with (
        patch.object(graph, "ChatAnthropic"),
        patch.object(graph, "create_supervisor") as mock_create_supervisor,
    ):
        mock_workflow = MagicMock()
        mock_compiled = MagicMock()
        mock_workflow.compile.return_value = mock_compiled
        mock_create_supervisor.return_value = mock_workflow

        result = graph.build_agent()

        assert result is mock_compiled
