"""Free, non-LLM pre-check that runs before even the AI agent topic gate that uses regex pattern matching costs nothing, so a message this rejects never reaches a paid API call at all.
"""

import re

_MAX_MESSAGE_LENGTH = 1000  # generous character-cap for a real F1 question

_SHELL_PATTERN = re.compile(
    r"(\$\(|`|&&|\|\||;\s*\S|>\s*/|<\s*/|\brm\s+-rf\b|\bsudo\b|\bchmod\b|\bchown\b"
    r"|\bcurl\b|\bwget\b|/etc/passwd|\b(?:bash|zsh|powershell|cmd\.exe)\b)",
    re.IGNORECASE,
)

_INJECTION_PATTERN = re.compile(
    r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions\b"
    r"|\byou\s+are\s+now\b"
    r"|\bdeveloper\s+mode\b"
    r"|\bjailbreak\b"
    r"|\b(?:reveal|print|show)\s+(?:your\s+)?(?:system\s+prompt|instructions)\b"
    r"|\bsystem\s+prompt\b"
    r"|\bpretend\s+(?:you\s+are|to\s+be)\b",
    re.IGNORECASE,
)

_IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b"
)

_IPV6_GROUP_PATTERN = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b")
_HEX_LETTER_PATTERN = re.compile(r"[a-fA-F]")

_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)


def _contains_ipv6(message: str) -> bool:
    if "::" in message:
        return True
    match = _IPV6_GROUP_PATTERN.search(message)
    return bool(match and _HEX_LETTER_PATTERN.search(match.group()))


def check_input_safety(message: str) -> dict:
    """Returns {"safe": bool, "reason": str | None}.

    `reason` is one of "too_long", "shell_command", "prompt_injection",
    "ip_address", "url" when unsafe - checked in that order, so a message
    tripping more than one pattern reports whichever is checked first.
    The length check runs first since it's a plain len() call - no need
    to run any regex over a huge pasted blob before rejecting it.
    """
    if len(message) > _MAX_MESSAGE_LENGTH:
        return {"safe": False, "reason": "too_long"}
    if _SHELL_PATTERN.search(message):
        return {"safe": False, "reason": "shell_command"}
    if _INJECTION_PATTERN.search(message):
        return {"safe": False, "reason": "prompt_injection"}
    if _IPV4_PATTERN.search(message) or _contains_ipv6(message):
        return {"safe": False, "reason": "ip_address"}
    if _URL_PATTERN.search(message):
        return {"safe": False, "reason": "url"}
    return {"safe": True, "reason": None}
