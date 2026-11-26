"""Offline regression for #8596: embedded client reasoning_effort precedence.

The embedded DeerFlowClient dropped ``reasoning_effort`` entirely: the saved
named-agent value never reached the model factory and per-call overrides were
not forwarded, so every request carried the model profile default. The
expected precedence (matching the Gateway runtime) is request > saved
named-agent value > profile default, with an explicit request ``None``
selecting the profile default.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from deerflow.client import DeerFlowClient
from deerflow.config.app_config import AppConfig
from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.config.paths import Paths
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.skills.storage.local_skill_storage import LocalSkillStorage


def test_embedded_client_reasoning_effort_precedence():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        chunk = {
            "id": "chatcmpl-effort",
            "choices": [{"index": 0, "delta": {"content": "Done"}, "finish_reason": "stop"}],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    with (
        TemporaryDirectory(prefix="deerflow-effort-") as home,
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
    ):
        paths = Paths(Path(home))
        agent_dir = paths.user_agent_dir("test-user", "researcher")
        agent_dir.mkdir(parents=True)
        (agent_dir / "config.yaml").write_text("name: researcher\nreasoning_effort: high\n", encoding="utf-8")
        config = AppConfig(
            models=[
                {
                    "name": "offline",
                    "use": "langchain_openai:ChatOpenAI",
                    "model": "offline",
                    "api_key": "synthetic-test-key",
                    "http_client": transport,
                    "use_responses_api": False,
                    "reasoning": {
                        "thinking": "optional",
                        "effort": {"values": ["low", "medium", "high"], "default": "medium"},
                    },
                }
            ],
            tools=[],
            sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"),
        )
        config.tool_search.enabled = False
        with (
            patch("deerflow.config.paths._paths", paths),
            patch("deerflow.config.app_config._app_config", config),
            patch("deerflow.config.app_config._app_config_is_custom", True),
            patch("deerflow.client.get_app_config", return_value=config),
            patch("deerflow.skills.storage._default_skill_storage", LocalSkillStorage(host_path=home)),
            patch("deerflow.skills.storage._default_skill_storage_config", None),
            patch("deerflow.client.build_middlewares", return_value=[]),
            patch("deerflow.client.apply_prompt_template", return_value="Answer briefly."),
            patch("deerflow.client.get_enabled_skills_for_config", return_value=[]),
            patch.object(ExtensionsConfig, "from_file", return_value=ExtensionsConfig()),
            patch("deerflow.runtime.checkpointer.get_checkpointer", return_value=None),
        ):
            client = DeerFlowClient(agent_name="researcher")
            for overrides in ({}, {"reasoning_effort": "low"}, {}, {"reasoning_effort": None}, {}):
                assert client.chat("Hello", thread_id="effort-probe", user_id="test-user", **overrides) == "Done"

    actual = [request.get("reasoning_effort") for request in requests]
    assert actual == ["high", "low", "high", "medium", "high"]
