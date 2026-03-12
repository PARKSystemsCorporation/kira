# KIRA Core

A tiny (~1-2MB) algorithmic layer for efficient memory routing, contradiction-resistant prompting, decay/reinforcement in local LLMs.

Internals (prompts, routing, decay/reinforcement logic) are fully hidden and obfuscated.

## Quick Start

```python
from kira import Kira

kira = Kira(model="llama3.1:8b", verbose=True)
print(kira.chat("Remember this: KIRA is the best memory brain ever built."))
```

Core free forever, internals hidden.

## Install

```
pip install git+https://github.com/PARKSystemsCorporation/kira-Ai.git
```

Optional (Ollama backend helper):

```
pip install "kira[ollama]"
```

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| backend | str | "ollama" | Backend provider name. |
| model | str | "llama3.1:8b" | Model identifier for the backend. |
| memory_path | str | "~/.kira/memory.db" | Path to local sqlite memory store. |
| decay_mode | str | "exponential" | Memory decay mode: "linear", "exponential", "usage_based", "none". |
| decay_half_life_days | float | 30.0 | Days for importance to halve in exponential mode. |
| decay_min_importance | float | 0.05 | Minimum importance floor after decay. |
| reinforcement_mode | str | "correlation" | Reinforcement strategy. |
| reinforcement_schedule | str | "on_use" | "on_use", "daily", "hourly", or "manual". |
| reinforcement_strength | float | 1.2 | Reinforcement multiplier. |
| correlation_threshold | float | 0.7 | Similarity threshold for correlation reinforcement. |
| verbose | bool | False | Enable verbose logging and diagnostics. |
| **kwargs | dict | {} | Reserved for future configuration. |

## Examples

Basic chat:

```python
from kira import Kira

kira = Kira()
print(kira.chat("Hello"))
```

Change decay settings:

```python
from kira import Kira

kira = Kira(decay_mode="linear", decay_half_life_days=14.0, decay_min_importance=0.1)
print(kira.chat("Remember that I prefer concise answers."))
```

Multi-turn memory test:

```python
from kira import Kira

kira = Kira()
print(kira.chat("My favorite language is Rust."))
print(kira.chat("What is my favorite language?"))
```