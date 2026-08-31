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
