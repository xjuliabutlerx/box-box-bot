from unittest.mock import MagicMock, patch

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


def test_build_agent_passes_both_specialists():
    with (
        patch.object(graph, "ChatAnthropic"),
        patch.object(graph, "create_supervisor") as mock_create_supervisor,
    ):
        mock_create_supervisor.return_value = MagicMock()

        graph.build_agent()

        (agents,), _ = mock_create_supervisor.call_args
        names = {agent.name for agent in agents}
        assert names == {"stats_agent", "narrative_agent"}


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
