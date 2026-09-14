import datetime
from unittest.mock import patch

from box_box_bot.agent.time_context import current_date_context


def test_current_date_context_uses_todays_date():
    fake_today = datetime.date(2026, 8, 30)
    with patch("box_box_bot.agent.time_context.datetime") as mock_datetime:
        mock_datetime.date.today.return_value = fake_today
        result = current_date_context()

    assert "2026-08-30" in result
    assert "2026" in result


def test_current_date_context_mentions_relative_time_phrases():
    result = current_date_context()
    assert "this year" in result
    assert "the current season" in result


def test_current_date_context_instructs_default_season_scoping():
    # A vague question ("what's typical tire strategy at Monaco?") that
    # names no year at all must default to the current season, not spread
    # tool calls across several past seasons to build a "general" answer -
    # while leaving genuinely multi-season tools (all-time records, a
    # circuit's historical SC pattern) unaffected.
    fake_today = datetime.date(2026, 8, 30)
    with patch("box_box_bot.agent.time_context.datetime") as mock_datetime:
        mock_datetime.date.today.return_value = fake_today
        result = current_date_context()

    assert "doesn't name a season/year" in result
    assert "2026" in result
    assert "career/all-time records" in result
