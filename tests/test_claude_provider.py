"""Tests for native Anthropic Claude provider."""
import pytest
from unittest.mock import patch, MagicMock

from claude_provider import (
    ClaudeClient, ClaudeResponse, AVAILABLE_MODELS, CLAUDE_PRICING,
    TRANSIENT_STATUSES, ANTHROPIC_VERSION, PROMPT_CACHING_BETA,
    is_anthropic_provider,
)


class TestClaudeClientConfig:
    def test_unconfigured_when_no_key(self):
        c = ClaudeClient(api_key="")
        assert not c.is_configured

    def test_configured_with_real_key(self):
        c = ClaudeClient(api_key="sk-ant-real-key-12345")
        assert c.is_configured

    def test_placeholder_key_not_configured(self):
        c = ClaudeClient(api_key="sk-xxx-placeholder")
        assert not c.is_configured

    def test_default_model(self):
        c = ClaudeClient(api_key="sk-ant-test")
        assert c.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        c = ClaudeClient(api_key="sk-ant-test", model="claude-opus-4-7")
        assert c.model == "claude-opus-4-7"

    def test_strips_trailing_slash_from_base_url(self):
        c = ClaudeClient(api_key="k", base_url="https://aiprimetech.io/")
        assert c.base_url == "https://aiprimetech.io"

    def test_default_base_url_is_aiprimetech(self):
        # Matches ChatApp PhD project default
        c = ClaudeClient(api_key="k")
        assert c.base_url == "https://aiprimetech.io"

    def test_can_override_to_direct_anthropic(self):
        c = ClaudeClient(api_key="k", base_url="https://api.anthropic.com")
        assert c.base_url == "https://api.anthropic.com"


class TestBudgetTracking:
    def test_initial_budget(self):
        c = ClaudeClient(api_key="k", call_budget=10)
        assert c.remaining_budget == 10

    def test_reserve_decrements(self):
        c = ClaudeClient(api_key="k", call_budget=3)
        assert c._reserve_slot() is True
        assert c.remaining_budget == 2
        assert c._reserve_slot() is True
        assert c._reserve_slot() is True
        assert c._reserve_slot() is False  # exhausted
        assert c.remaining_budget == 0

    def test_refund_increments(self):
        c = ClaudeClient(api_key="k", call_budget=3)
        c._reserve_slot()
        assert c.remaining_budget == 2
        c._refund_slot()
        assert c.remaining_budget == 3

    def test_reset_budget(self):
        c = ClaudeClient(api_key="k", call_budget=5)
        for _ in range(3):
            c._reserve_slot()
        assert c.remaining_budget == 2
        c.reset_budget()
        assert c.remaining_budget == 5


class TestPricing:
    def test_all_models_have_pricing(self):
        for m in AVAILABLE_MODELS:
            assert m in CLAUDE_PRICING
            p = CLAUDE_PRICING[m]
            assert all(k in p for k in ("input", "output", "cache_read", "cache_write"))

    def test_cost_computation(self):
        # 1M input tokens of opus at $15
        cost = ClaudeClient._compute_cost("claude-opus-4-7", 1_000_000, 0)
        assert cost == 15.0

    def test_cache_read_cheaper(self):
        # cache_read should be ~10% of input cost
        cost_input = ClaudeClient._compute_cost("claude-sonnet-4-6", 1_000_000, 0, 0, 0)
        cost_cached = ClaudeClient._compute_cost("claude-sonnet-4-6", 0, 0, 1_000_000, 0)
        assert cost_cached < cost_input * 0.2  # at least 5x cheaper

    def test_zero_tokens_zero_cost(self):
        assert ClaudeClient._compute_cost("claude-haiku-4-5", 0, 0, 0, 0) == 0.0

    def test_haiku_cheapest_output(self):
        haiku_out = CLAUDE_PRICING["claude-haiku-4-5"]["output"]
        sonnet_out = CLAUDE_PRICING["claude-sonnet-4-6"]["output"]
        opus_out = CLAUDE_PRICING["claude-opus-4-7"]["output"]
        assert haiku_out < sonnet_out < opus_out


class TestApiCallMocked:
    def _mock_success(self, status=200, payload=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.text = ""
        mock_resp.json.return_value = payload or {
            "content": [{"type": "text", "text": "hello world"}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 0,
            },
        }
        return mock_resp

    def test_unconfigured_returns_error(self):
        c = ClaudeClient(api_key="")
        result = c.call("system", "user")
        assert not result.success
        assert "ANTHROPIC_API_KEY" in result.error

    def test_budget_exhausted_returns_error(self):
        c = ClaudeClient(api_key="sk-ant-test", call_budget=0)
        result = c.call("system", "user")
        assert not result.success
        assert "Budget exhausted" in result.error

    def test_successful_call_parses_response(self):
        c = ClaudeClient(api_key="sk-ant-test")
        with patch.object(c._session, "post", return_value=self._mock_success()):
            result = c.call("you are a helper", "say hello")
        assert result.success
        assert result.text == "hello world"
        assert result.input_tokens == 50
        assert result.output_tokens == 20
        assert result.cache_read_tokens == 100
        assert result.cost_usd > 0

    def test_request_includes_anthropic_headers(self):
        c = ClaudeClient(api_key="sk-ant-test")
        with patch.object(c._session, "post", return_value=self._mock_success()) as mp:
            c.call("system", "user")
            _, kwargs = mp.call_args
            headers = kwargs["headers"]
            assert headers["Authorization"] == "Bearer sk-ant-test"
            assert headers["anthropic-version"] == ANTHROPIC_VERSION
            assert headers["anthropic-beta"] == PROMPT_CACHING_BETA

    def test_system_prompt_marked_for_caching(self):
        c = ClaudeClient(api_key="sk-ant-test")
        with patch.object(c._session, "post", return_value=self._mock_success()) as mp:
            c.call("my system prompt", "user msg", cache_system=True)
            _, kwargs = mp.call_args
            body = kwargs["json"]
            assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
            assert body["system"][0]["text"] == "my system prompt"

    def test_cache_disabled(self):
        c = ClaudeClient(api_key="sk-ant-test")
        with patch.object(c._session, "post", return_value=self._mock_success()) as mp:
            c.call("sys", "user", cache_system=False)
            _, kwargs = mp.call_args
            body = kwargs["json"]
            assert "cache_control" not in body["system"][0]

    def test_json_mode_appends_instruction(self):
        c = ClaudeClient(api_key="sk-ant-test")
        with patch.object(c._session, "post", return_value=self._mock_success()) as mp:
            c.call("sys", "give me data", json_mode=True)
            _, kwargs = mp.call_args
            user_msg = kwargs["json"]["messages"][0]["content"]
            assert "JSON" in user_msg or "json" in user_msg

    def test_4xx_no_retry(self):
        c = ClaudeClient(api_key="sk-ant-test")
        bad = MagicMock(status_code=401, text="Invalid API key")
        with patch.object(c._session, "post", return_value=bad) as mp:
            result = c.call("s", "u")
            assert mp.call_count == 1  # no retry
            assert not result.success
            assert "401" in result.error
            # Budget should be refunded
            assert c.remaining_budget == c.call_budget

    def test_429_retries_then_succeeds(self):
        c = ClaudeClient(api_key="sk-ant-test")
        rate_limited = MagicMock(status_code=429, text="rate limited")
        success = self._mock_success()
        with patch.object(c._session, "post", side_effect=[rate_limited, success]):
            with patch("time.sleep"):  # skip backoff delay in tests
                result = c.call("s", "u")
            assert result.success
            assert result.attempts == 2

    def test_all_attempts_fail_refunds_budget(self):
        c = ClaudeClient(api_key="sk-ant-test", call_budget=5)
        rate_limited = MagicMock(status_code=503, text="overloaded")
        with patch.object(c._session, "post", return_value=rate_limited):
            with patch("time.sleep"):
                result = c.call("s", "u")
            assert not result.success
            assert result.attempts == 3
            assert c.remaining_budget == 5  # refunded


class TestTransientStatuses:
    def test_includes_429(self):
        assert 429 in TRANSIENT_STATUSES

    def test_includes_5xx(self):
        for code in (500, 502, 503, 504):
            assert code in TRANSIENT_STATUSES

    def test_excludes_4xx_validation(self):
        for code in (400, 401, 403, 404):
            assert code not in TRANSIENT_STATUSES


class TestProviderDetection:
    def test_recognizes_anthropic(self):
        assert is_anthropic_provider("anthropic")
        assert is_anthropic_provider("ANTHROPIC")
        assert is_anthropic_provider("claude")

    def test_rejects_other_providers(self):
        assert not is_anthropic_provider("openai")
        assert not is_anthropic_provider("deepseek")
        assert not is_anthropic_provider("")
