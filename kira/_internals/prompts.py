from __future__ import annotations

import base64
import zlib
from typing import Dict

COMPRESSED_PROMPTS = {
    "default": "eJxljs1uwkAMhO/7FPMAlAfghtoL4obEgaOzcYilZI3WTtqo4t0xP+2F22j8zXhOOoEqY787bFegAinOwyBnLg4yE3MK9S3eg3JmM7hi5FHrgqwB//g6pcM0sG3SB47GuFSdpeX27w7pUHngOZrWwew6FP134mOnEIsYW3xdwWjBErtaDc7R08wPqI7kogXU6BSxe/K15N769aQ7aqpkcg6V3dDwoqWF9++7YvfnU23S78u7phsOvGFV",
}


def _get_prompt(key: str) -> str:
    return zlib.decompress(base64.b64decode(COMPRESSED_PROMPTS[key])).decode("utf-8")


def _obfuscate_strings(strings: Dict[str, str]) -> Dict[str, str]:
    return {
        key: base64.b64encode(zlib.compress(value.encode("utf-8"))).decode("utf-8")
        for key, value in strings.items()
    }