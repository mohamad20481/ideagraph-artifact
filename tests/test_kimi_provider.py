"""Unit tests for the Kimi (Moonshot AI) provider wiring.

These tests do NOT make real network calls — they verify configuration,
client construction, failover mapping, JSON-mode dispatch, and the
temperature-coercion logic that handles Moonshot's per-model quirks.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import config


# ── Config wiring ────────────────────────────────────────────────────────

class TestConfigWiring:
    def test_kimi_in_supported_providers(self):
        assert "kimi" in config.SUPPORTED_PROVIDERS

    def test_kimi_default_model_set(self):
        assert "kimi" in config._DEFAULT_MODELS
        # Must reference a real model ID, not a placeholder
        assert config._DEFAULT_MODELS["kimi"]
        assert isinstance(config._DEFAULT_MODELS["kimi"], str)

    def test_kimi_cost_rates_present(self):
        assert "kimi" in config.COST_RATES
        rates = config.COST_RATES["kimi"]
        assert "input" in rates and "output" in rates
        assert rates["input"] > 0 and rates["output"] > 0

    def test_kimi_base_url_attribute_present(self):
        # Must be an attribute on the module (read at agent-construction time)
        assert hasattr(config, "KIMI_BASE_URL")
        assert config.KIMI_BASE_URL.startswith("https://")

    def test_kimi_api_key_attribute_present(self):
        # Attribute exists (may be empty string in CI; we don't assert non-empty)
        assert hasattr(config, "KIMI_API_KEY")

    def test_kimi_api_key_falls_back_to_moonshot_env(self):
        # The fallback `os.getenv("MOONSHOT_API_KEY")` must be present in
        # the source so users can use either env-var name
        with open("config.py", encoding="utf-8") as f:
            src = f.read()
        assert "MOONSHOT_API_KEY" in src


# ── base_agent client construction ───────────────────────────────────────

class TestClientConstruction:
    def test_kimi_client_uses_correct_base_url(self):
        from agents.base_agent import BaseAgent
        client = BaseAgent._build_client_for_provider("kimi")
        # The OpenAI SDK normalizes base_url with a trailing slash
        assert "moonshot.ai" in str(client.base_url) or \
               "moonshot.cn" in str(client.base_url)

    def test_kimi_client_uses_kimi_api_key(self):
        # Patch config.KIMI_API_KEY temporarily
        with patch.object(config, "KIMI_API_KEY", "sk-test-kimi-12345"):
            from agents.base_agent import BaseAgent
            client = BaseAgent._build_client_for_provider("kimi")
            assert client.api_key == "sk-test-kimi-12345"

    def test_default_client_branch_unchanged(self):
        # Sanity: deepseek client builder still works after our additions
        from agents.base_agent import BaseAgent
        client = BaseAgent._build_client_for_provider("deepseek")
        assert "deepseek" in str(client.base_url)


# ── Failover map ─────────────────────────────────────────────────────────

class TestFailoverMap:
    def test_kimi_appears_in_failover_order(self):
        # Read the source so we don't have to instantiate / hit the network
        with open("agents/base_agent.py", encoding="utf-8") as f:
            src = f.read()
        # The key map must reference kimi → KIMI_API_KEY (whitespace tolerant)
        import re
        assert re.search(r'"kimi"\s*:\s*config\.KIMI_API_KEY', src) is not None
        # And kimi must appear in the fallback_order list
        assert re.search(r'fallback_order\s*=\s*\[[^\]]*"kimi"', src) is not None

    def test_failover_picks_kimi_when_key_present(self):
        from agents.base_agent import BaseAgent
        agent = BaseAgent()
        # Pretend we're on openai but only kimi has a key
        with patch.object(config, "PROVIDER", "openai"), \
             patch.object(config, "DEEPSEEK_API_KEY", ""), \
             patch.object(config, "KIMI_API_KEY", "sk-test-fallback"), \
             patch.object(config, "GEMINI_API_KEY", ""), \
             patch.object(config, "GROQ_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "AZURE_API_KEY", ""):
            failover = agent._get_failover_provider()
            assert failover == "kimi"


# ── speed_optimizer integration ──────────────────────────────────────────

class TestSpeedOptimizer:
    def test_kimi_in_at_least_one_tier(self):
        from speed_optimizer import TIER_MODELS
        all_provs = []
        for tier_list in TIER_MODELS.values():
            all_provs.extend(p for p, _ in tier_list)
        assert "kimi" in all_provs, \
            "kimi should appear in TIER_MODELS so stage routing can use it"

    def test_kimi_in_has_key_lookup(self):
        from speed_optimizer import _has_key
        with patch.object(config, "KIMI_API_KEY", "sk-test"):
            assert _has_key("kimi") is True
        with patch.object(config, "KIMI_API_KEY", ""):
            assert _has_key("kimi") is False

    def test_route_for_stage_can_pick_kimi(self):
        # When kimi is the only configured key, stage routing must pick it
        from speed_optimizer import route_for_stage
        with patch.object(config, "DEEPSEEK_API_KEY", ""), \
             patch.object(config, "ANTHROPIC_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.object(config, "GEMINI_API_KEY", ""), \
             patch.object(config, "GROQ_API_KEY", ""), \
             patch.object(config, "AZURE_API_KEY", ""), \
             patch.object(config, "KIMI_API_KEY", "sk-test"):
            provider, model = route_for_stage("ideation")
            assert provider == "kimi"
            assert "moonshot" in model or "kimi" in model


# ── Temperature coercion (the load-bearing fix) ──────────────────────────

class TestTemperatureCoercion:
    def test_source_contains_kimi_temperature_block(self):
        # The coercion logic must be present in _call so probes/agents that
        # pass temperature=0.0 don't get hard-rejected by Moonshot.
        with open("agents/base_agent.py", encoding="utf-8") as f:
            src = f.read()
        assert 'PROVIDER == "kimi"' in src
        # Two coercion paths: reasoning model → 1.0; non-reasoning → max(0.3, T)
        assert "kimi-k2" in src
        assert "max(0.3" in src

    def test_json_mode_supports_kimi(self):
        with open("agents/base_agent.py", encoding="utf-8") as f:
            src = f.read()
        # JSON mode dispatch list must include kimi
        assert '"kimi"' in src
        # Spot-check the response_format line is present
        assert 'response_format' in src and 'json_object' in src


# ── End-to-end smoke (no network) ────────────────────────────────────────

class TestEndToEndShape:
    def test_pipeline_does_not_crash_when_provider_kimi(self):
        # We don't actually run the pipeline, but constructing an agent
        # with PROVIDER=kimi must succeed (no exception, real client built).
        with patch.object(config, "PROVIDER", "kimi"), \
             patch.object(config, "MODEL", "moonshot-v1-32k"), \
             patch.object(config, "KIMI_API_KEY", "sk-test-12345"):
            from agents.base_agent import BaseAgent
            agent = BaseAgent()
            assert agent.client is not None
            # Client should target Moonshot, not DeepSeek
            assert "moonshot" in str(agent.client.base_url)
