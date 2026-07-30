# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the aggregated web_search builtin tool.

RFC-0028: 聚合 Web 搜索内置工具

所有用例都**不打真实上游**——通过注入 httpx MockTransport 断言请求侧，
因此 CI 不需要任何 API Key。
"""

import json
import logging
from pathlib import Path

import httpx
import pytest

from nexau.archs.tool.builtin.web_tools import aggregated_search as agg
from nexau.archs.tool.builtin.web_tools import web_search

_SEARCH_ENV_PREFIX = "SEARCH_"
_LEGACY_ENV = "SERPER_API_KEY"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个用例都从干净的环境开始，避免开发机上的真实 Key 干扰断言。"""
    for key in list(agg.os.environ):
        if key.startswith(_SEARCH_ENV_PREFIX) or key == _LEGACY_ENV:
            monkeypatch.delenv(key, raising=False)
    agg._provider_cache.clear()


class _Recorder:
    """记录最后一次发出的请求，并按需返回预置响应。"""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.calls = 0
        self.last_url = ""
        self.last_headers: dict[str, str] = {}
        self.last_body: dict[str, object] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.last_url = str(request.url)
        self.last_headers = dict(request.headers)
        if request.content:
            parsed = json.loads(request.content)
            if isinstance(parsed, dict):
                self.last_body = {str(k): v for k, v in parsed.items()}
        return httpx.Response(self.status_code, json=self.payload)


def _install(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    """把 httpx.Client 换成走 MockTransport 的版本，并禁用退避 sleep。"""
    real_client = httpx.Client

    def _factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs.pop("transport", None)
        timeout = kwargs.get("timeout")
        return real_client(
            transport=httpx.MockTransport(recorder.handler),
            timeout=timeout if isinstance(timeout, (int, float, httpx.Timeout)) else None,
        )

    monkeypatch.setattr(agg.httpx, "Client", _factory)
    monkeypatch.setattr(agg.time, "sleep", lambda _seconds: None)


def _set(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)


# --------------------------------------------------------------------- 入参校验


class TestQueryValidation:
    """空 query 直接短路，不产生任何上游请求。"""

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_empty_query_short_circuits(self, monkeypatch: pytest.MonkeyPatch, query: str) -> None:
        recorder = _Recorder({})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="k")

        result = web_search(query=query)

        assert result["error"]["type"] == "INVALID_QUERY"
        assert recorder.calls == 0, "空 query 不应该打上游"


# ----------------------------------------------------------------- 配置类错误


class TestConfiguration:
    def test_missing_api_key_reports_variable_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """缺 Key 时报错必须点名要设哪个环境变量。"""
        result = web_search(query="x")

        assert result["error"]["type"] == "WEB_SEARCH_CONFIG_ERROR"
        assert "SEARCH_API_KEY" in result["error"]["message"]

    def test_unknown_provider_lists_valid_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set(monkeypatch, SEARCH_PROVIDER="Bing", SEARCH_API_KEY="k")

        result = web_search(query="x")

        assert result["error"]["type"] == "WEB_SEARCH_CONFIG_ERROR"
        assert "Serper" in result["error"]["message"]

    def test_legacy_serper_api_key_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """向后兼容：只设旧的 SERPER_API_KEY 也要能正常工作。"""
        recorder = _Recorder({"organic": [{"title": "T", "link": "https://e.com"}]})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SERPER_API_KEY="legacy-key")

        result = web_search(query="x")

        assert result.get("error") is None
        assert result["provider"] == "Serper"
        assert recorder.last_headers.get("x-api-key") == "legacy-key"


# ------------------------------------------------------------ 各服务商请求契约


class TestProviderRequestContracts:
    """断言发给各家上游的 URL / 鉴权头 / 请求体符合其官方文档。"""

    def test_serper_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        web_search(query="q", num_results=3, sites="arxiv.org", time_range="OneWeek")

        assert recorder.last_url.endswith("/search")
        assert recorder.last_headers.get("x-api-key") == "sk"
        assert recorder.last_body["num"] == 3
        assert "site:arxiv.org" in str(recorder.last_body["q"])
        assert recorder.last_body["tbs"] == "qdr:w"

    def test_seed_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"Result": {"WebResults": []}})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Seed", SEARCH_API_KEY="sk")

        web_search(query="q", num_results=3, authority_only=True, industry="gov")

        assert recorder.last_url.endswith("/search_api/web_search")
        assert recorder.last_headers.get("authorization") == "Bearer sk"
        assert recorder.last_body["Count"] == 3
        filters = recorder.last_body["Filter"]
        assert isinstance(filters, dict)
        assert filters["AuthInfoLevel"] == 1
        assert filters["Industry"] == "gov"

    def test_baidu_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"references": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Baidu", SEARCH_API_KEY="sk")

        web_search(query="q", num_results=3, sites="people.com.cn")

        assert recorder.last_url.endswith("/v2/ai_search/web_search")
        assert recorder.last_headers.get("authorization") == "Bearer sk"
        assert recorder.last_body["search_source"] == "baidu_search_v2"
        search_filter = recorder.last_body["search_filter"]
        assert isinstance(search_filter, dict)
        assert search_filter["match"] == {"site": ["people.com.cn"]}

    def test_xiaobei_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"status": "search_done", "results": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="XiaoBei", SEARCH_API_KEY="sk")

        web_search(query="q", num_results=3, search_engine="baidu")

        assert recorder.last_url.endswith("/tools/web_research")
        assert recorder.last_headers.get("x-api-key") == "sk"
        options = recorder.last_body["options"]
        assert isinstance(options, dict)
        assert options["engines"] == ["baidu"]


# ------------------------------------------------------------- 参数降级与告警


class TestUnsupportedParams:
    def test_unsupported_param_warns_but_succeeds(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """服务商不支持的参数：调用照常成功，但必须留痕。"""
        recorder = _Recorder({"organic": [{"title": "T", "link": "https://e.com"}]})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        with caplog.at_level(logging.WARNING):
            result = web_search(query="q", authority_only=True)

        assert result.get("error") is None, "不支持的参数不应导致调用失败"
        assert any("authority_only" in record.message for record in caplog.records)

    def test_engine_choice_warns_on_non_aggregator(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        with caplog.at_level(logging.WARNING):
            web_search(query="q", search_engine="baidu")

        assert any("search_engine" in record.message for record in caplog.records)


# --------------------------------------------------------------- 参数优先级


class TestParameterResolution:
    def test_env_provides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_NUM_RESULTS="4",
        )

        web_search(query="q")

        assert recorder.last_body["num"] == 4

    def test_explicit_argument_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """入参优先于环境变量——模型显式指定时必须以它为准。"""
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_NUM_RESULTS="4",
        )

        web_search(query="q", num_results=7)

        assert recorder.last_body["num"] == 7

    def test_explicit_value_equal_to_default_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """模型回传的默认值也算"显式传参"，不能被环境变量盖掉。

        这是 None 哨兵存在的理由：实测模型会把 schema 默认值原样回填。
        """
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_NUM_RESULTS="4",
        )

        web_search(query="q", num_results=agg.DEFAULT_NUM_RESULTS)

        assert recorder.last_body["num"] == agg.DEFAULT_NUM_RESULTS

    def test_malformed_env_falls_back_with_warning(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_NUM_RESULTS="abc",
        )

        with caplog.at_level(logging.WARNING):
            result = web_search(query="q")

        assert result.get("error") is None, "环境变量写错不应打挂检索"
        assert recorder.last_body["num"] == agg.DEFAULT_NUM_RESULTS
        assert any("SEARCH_NUM_RESULTS" in record.message for record in caplog.records)

    def test_param_env_name_strips_redundant_prefix(self) -> None:
        assert agg.param_env_name("content_format") == "SEARCH_CONTENT_FORMAT"
        assert agg.param_env_name("search_engine") == "SEARCH_ENGINE"

    def test_param_env_names_never_collide_with_provider_env(self) -> None:
        """参数级变量名不得与服务商级变量撞名，否则会互相覆盖。"""
        param_envs = {agg.param_env_name(name) for name in agg.PARAM_DEFAULTS}

        assert not (param_envs & agg.PROVIDER_ENV_KEYS)


# ------------------------------------------------------------------ 错误分类


class TestErrorClassification:
    def test_server_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({}, status_code=503)
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="3",
        )

        result = web_search(query="q")

        assert result["error"]["type"] == "WEB_SEARCH_FAILED"
        assert recorder.calls == 3, "5xx 应按 SEARCH_MAX_RETRIES 重试"

    def test_client_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({}, status_code=403)
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="3",
        )

        result = web_search(query="q")

        assert result["error"]["type"] == "WEB_SEARCH_FAILED"
        assert recorder.calls == 1, "4xx 是确定性错误，不该重试"

    def test_seed_business_error_inside_http_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """豆包失败时也返回 HTTP 200，错误藏在 ResponseMetadata.Error。"""
        recorder = _Recorder(
            {
                "ResponseMetadata": {
                    "Error": {
                        "CodeN": 10406,
                        "Code": "FreeQuotaExhausted",
                        "Message": "quota",
                    }
                },
                "Result": None,
            }
        )
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Seed",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="3",
        )

        result = web_search(query="q")

        assert result["error"]["type"] == "WEB_SEARCH_CONFIG_ERROR"
        assert recorder.calls == 1, "额度用尽属于配置类错误，重试无益"

    def test_seed_retryable_business_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder(
            {
                "ResponseMetadata": {"Error": {"CodeN": 10500, "Code": "InnerError", "Message": "boom"}},
                "Result": None,
            }
        )
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Seed",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="3",
        )

        result = web_search(query="q")

        assert result["error"]["type"] == "WEB_SEARCH_FAILED"
        assert recorder.calls == 3, "服务端内部错误应重试"


# -------------------------------------------------------------------- 返回结构


class TestOutputFormat:
    def test_success_shape_matches_builtin_convention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """返回结构须与既有内置搜索工具一致，前端与 prompt 无需改动。"""
        recorder = _Recorder({"organic": [{"title": "Python", "link": "https://python.org", "snippet": "s"}]})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        result = web_search(query="python")

        assert result.get("error") is None
        assert "Python" in result["content"]
        assert result["returnDisplay"]
        assert result["sources"][0]["link"] == "https://python.org"
        assert result["provider"] == "Serper"

    def test_zero_results_is_success_not_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """搜不到 ≠ 出错——让模型知道"确实没有"，而不是当故障重试。"""
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        result = web_search(query="obscure")

        assert result.get("error") is None
        assert result["sources"] == []
        assert "No results" in result["content"]

    def test_max_content_chars_truncates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"organic": [{"title": "T", "link": "https://e.com", "snippet": "x" * 900}]})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        result = web_search(query="q", max_content_chars=100)

        assert len(result["sources"][0]["snippet"]) <= 110


# ------------------------------------------------------ tool schema 与实现一致


class TestSchemaConsistency:
    """`.tool.yaml` 与 Python 签名是一式两份，必然漂移——用测试钉住。"""

    def _schema(self) -> dict[str, object]:
        """按 runtime 实际使用的方式定位 schema，避免测试与实现走两条路径。"""
        import yaml

        from nexau.archs.main_sub.config.config import (
            _BUILTIN_TOOL_SCHEMA_ROOT,
            _resolve_config_path,
        )

        path = _resolve_config_path(f"{_BUILTIN_TOOL_SCHEMA_ROOT}/web_search.tool.yaml", Path.cwd())
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        assert isinstance(loaded, dict)
        return {str(k): v for k, v in loaded.items()}

    def test_parameter_sets_match(self) -> None:
        import inspect

        schema = self._schema()
        input_schema = schema["input_schema"]
        assert isinstance(input_schema, dict)
        properties = input_schema["properties"]
        assert isinstance(properties, dict)
        signature = inspect.signature(web_search)

        assert set(properties) == set(signature.parameters)

    def test_defaults_match(self) -> None:
        schema = self._schema()
        input_schema = schema["input_schema"]
        assert isinstance(input_schema, dict)
        properties = input_schema["properties"]
        assert isinstance(properties, dict)

        for name, spec in properties.items():
            if name == "query":
                continue
            assert isinstance(spec, dict)
            assert spec.get("default") == agg.PARAM_DEFAULTS[str(name)], f"{name} 的 YAML default 与 PARAM_DEFAULTS 不一致"

    def test_signature_defaults_are_sentinels(self) -> None:
        """非必填参数的签名默认值必须是 None 哨兵（见 RFC-0028 参数优先级）。"""
        import inspect

        signature = inspect.signature(web_search)

        for name, parameter in signature.parameters.items():
            if name == "query":
                continue
            assert parameter.default is None, f"{name} 的签名默认值应为 None 哨兵"


# ------------------------------------------------------------ 结果归一与边界


class TestResultNormalization:
    """四家的返回结构差异很大，归一后必须对上层呈现同一套字段。"""

    def test_seed_prefers_summary_over_snippet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """豆包文档明确 Snippet 不建议喂大模型，应优先取 Summary。"""
        recorder = _Recorder(
            {
                "Result": {
                    "WebResults": [
                        {
                            "Title": "T",
                            "Url": "https://e.com",
                            "Snippet": "短摘要",
                            "Summary": "长摘要" * 10,
                            "PublishTime": "2026-07-30T10:00:00+08:00",
                            "SiteName": "站点",
                            "AuthInfoDes": "非常权威",
                        }
                    ]
                }
            }
        )
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Seed", SEARCH_API_KEY="sk")

        result = web_search(query="q")

        source = result["sources"][0]
        assert source["snippet"].startswith("长摘要")
        assert source["authority"] == "非常权威"

    def test_seed_image_search_uses_image_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"Result": {"ImageResults": [{"Title": "P", "Url": "https://i.com/a.png"}]}})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Seed", SEARCH_API_KEY="sk")

        result = web_search(query="q", search_type="images")

        assert recorder.last_body["SearchType"] == "image"
        assert result["sources"][0]["link"] == "https://i.com/a.png"

    def test_serper_places_builds_map_link(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serper 的 places 结果没有 link 字段，只有 cid，需要拼出可点链接。"""
        recorder = _Recorder(
            {
                "places": [
                    {
                        "title": "Cafe",
                        "cid": "123",
                        "address": "Somewhere",
                        "category": "Coffee",
                        "rating": 4.5,
                        "ratingCount": 10,
                    }
                ]
            }
        )
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        result = web_search(query="cafe", search_type="places")

        source = result["sources"][0]
        assert source["link"] == "https://www.google.com/maps?cid=123"
        assert "Somewhere" in source["snippet"]
        assert "4.5" in source["snippet"]

    def test_baidu_drops_junk_source_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """百度实测 web_anchor 常为空串、website 偶尔是字面量"无"，都应视作缺失。"""
        recorder = _Recorder(
            {
                "references": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "snippet": "s",
                        "web_anchor": "",
                        "website": "无",
                    }
                ]
            }
        )
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Baidu", SEARCH_API_KEY="sk")

        result = web_search(query="q")

        assert result["sources"][0]["source"] is None

    def test_baidu_blocks_hosts_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """百度上游 block_websites 实测无效，必须本地兜底过滤。"""
        recorder = _Recorder(
            {
                "references": [
                    {"title": "A", "url": "https://spam.com/1", "snippet": "s"},
                    {"title": "B", "url": "https://good.com/2", "snippet": "s"},
                ]
            }
        )
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Baidu", SEARCH_API_KEY="sk")

        result = web_search(query="q", block_hosts="spam.com")

        links = [item["link"] for item in result["sources"]]
        assert links == ["https://good.com/2"]

    def test_xiaobei_reports_downstream_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder(
            {
                "status": "complete",
                "results": [
                    {
                        "rank": 1,
                        "title": "T",
                        "url": "https://e.com",
                        "snippet": "s",
                        "engine": "baidu",
                    }
                ],
            }
        )
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="XiaoBei", SEARCH_API_KEY="sk")

        result = web_search(query="q")

        assert result["sources"][0]["engine"] == "baidu"

    def test_xiaobei_empty_with_engine_errors_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """上游抖动（0 条 + engine_errors）是临时故障，不能当"搜不到"返回。"""
        recorder = _Recorder(
            {
                "status": "search_done",
                "results": [],
                "engine_errors": [{"engine": "google", "error": "timeout"}],
            }
        )
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="XiaoBei",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="2",
        )

        result = web_search(query="q")

        assert result["error"]["type"] == "WEB_SEARCH_FAILED"
        assert recorder.calls == 2

    def test_xiaobei_search_failed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder(
            {
                "status": "search_failed",
                "results": [],
                "failure_reason": "all_engines_error",
            }
        )
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="XiaoBei",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="1",
        )

        result = web_search(query="q")

        assert result["error"]["type"] == "WEB_SEARCH_FAILED"

    def test_xiaobei_scrape_failure_falls_back_to_snippet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """抓正文失败时降级回搜索摘要，而不是返回空内容。"""
        recorder = _Recorder(
            {
                "status": "failed",
                "results": [
                    {
                        "rank": 1,
                        "title": "T",
                        "url": "https://e.com",
                        "snippet": "回落摘要",
                        "engine": "bing",
                        "scrape": {"status": "failed", "markdown": ""},
                    }
                ],
            }
        )
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="XiaoBei", SEARCH_API_KEY="sk")

        result = web_search(query="q", full_content=True)

        assert result["sources"][0]["snippet"] == "回落摘要"


# ------------------------------------------------------------------ 辅助函数


class TestHelpers:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://a.example.com:8443/x", "a.example.com"),
            ("not-a-url", ""),
            ("", ""),
        ],
    )
    def test_host_of(self, url: str, expected: str) -> None:
        assert agg._host_of(url) == expected

    def test_host_matches_subdomain(self) -> None:
        assert agg._host_matches("bj.people.com.cn", "people.com.cn")
        assert not agg._host_matches("notpeople.com.cn", "people.com.cn")

    def test_site_operators_compose(self) -> None:
        compiled = agg._apply_site_operators("q", ["a.com", "b.com"], ["c.com"])

        assert "(site:a.com OR site:b.com)" in compiled
        assert "-site:c.com" in compiled

    def test_clean_date_drops_epoch_placeholder(self) -> None:
        assert agg._clean_date("1970-01-01T00:00:00Z") is None
        assert agg._clean_date("2026-07-30") == "2026-07-30"

    def test_custom_range_parsing(self) -> None:
        options = agg.SearchOptions(time_range="2026-01-01..2026-03-31")

        assert options.parse_custom_range() == ("2026-01-01", "2026-03-31")

    def test_serper_compiles_custom_date_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        web_search(query="q", time_range="2026-01-01..2026-03-31")

        assert recorder.last_body["tbs"] == "cdr:1,cd_min:1/1/2026,cd_max:3/31/2026"

    def test_serper_ignores_malformed_date_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """时效格式写错只忽略该参数，不该让整次检索失败。"""
        recorder = _Recorder({"organic": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Serper", SEARCH_API_KEY="sk")

        result = web_search(query="q", time_range="去年..今年")

        assert result.get("error") is None
        assert "tbs" not in recorder.last_body

    def test_baidu_one_day_uses_page_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """百度的 recency 枚举没有 day，OneDay 需退到 page_time 表达。"""
        recorder = _Recorder({"references": []})
        _install(monkeypatch, recorder)
        _set(monkeypatch, SEARCH_PROVIDER="Baidu", SEARCH_API_KEY="sk")

        web_search(query="q", time_range="OneDay")

        assert "search_recency_filter" not in recorder.last_body
        search_filter = recorder.last_body["search_filter"]
        assert isinstance(search_filter, dict)
        assert "page_time" in str(search_filter)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("yes", True), ("0", False), ("TRUE", True), ("off", False)],
    )
    def test_bool_env_parsing(self, raw: str, expected: bool) -> None:
        assert agg._coerce_env_value(raw, False, "SEARCH_X") is expected

    def test_engine_list_parsing_is_lenient(self) -> None:
        """环境变量与模型入参的格式都不完全可控，解析要宽松。"""
        for raw in ["google|bing", "google,bing", '["google","bing"]', "GOOGLE BING"]:
            options = agg.SearchOptions(search_engine=raw)

            assert options.split_engines() == ["google", "bing"], raw


# ------------------------------------------------------- 重试次数与传输层故障


class TestRetryBudgetBoundaries:
    """`SEARCH_MAX_RETRIES` 语义是**总尝试次数**，必须钳到 >=1。

    未钳制时 `range(0)` 会让循环体一次都不执行、`_do_search` 永不被调用，
    工具在不发出任何请求的情况下 100% 失败——而"重试次数=0"是很自然的误配。
    """

    @pytest.mark.parametrize("raw", ["0", "-1"])
    def test_non_positive_budget_still_sends_one_request(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        recorder = _Recorder({"organic": [{"title": "T", "link": "https://e.com"}]})
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES=raw,
        )

        result = web_search(query="q")

        assert recorder.calls == 1, f"SEARCH_MAX_RETRIES={raw} 不应导致零请求"
        assert result.get("error") is None

    def test_budget_of_one_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _Recorder({}, status_code=503)
        _install(monkeypatch, recorder)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="1",
        )

        result = web_search(query="q")

        assert recorder.calls == 1
        assert result["error"]["type"] == "WEB_SEARCH_FAILED"


class TestTransportFailures:
    """传输层异常（超时/断连）是最常见的真实故障，MockTransport 恒返回响应，
    必须显式让 handler 抛异常才能覆盖到这些分支。"""

    @staticmethod
    def _raising(exc: Exception, counter: dict[str, int]):
        def handler(_request: httpx.Request) -> httpx.Response:
            counter["calls"] += 1
            raise exc

        return handler

    @pytest.mark.parametrize(
        ("exc", "expected_text"),
        [
            (httpx.ConnectTimeout("boom"), "Connection timeout"),
            (httpx.ReadTimeout("boom"), "Request timeout"),
        ],
    )
    def test_timeout_is_retried_and_reported(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: Exception,
        expected_text: str,
    ) -> None:
        counter = {"calls": 0}
        real_client = httpx.Client

        def _factory(*_args: object, **_kwargs: object) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(self._raising(exc, counter)))

        monkeypatch.setattr(agg.httpx, "Client", _factory)
        monkeypatch.setattr(agg.time, "sleep", lambda _s: None)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="Serper",
            SEARCH_API_KEY="sk",
            SEARCH_MAX_RETRIES="3",
        )

        result = web_search(query="q")

        assert counter["calls"] == 3, "超时应按预算重试"
        assert result["error"]["type"] == "WEB_SEARCH_FAILED"
        assert expected_text in result["error"]["message"]


# --------------------------------------------------- XiaoBei 异步抓正文轮询


class _SequenceRecorder:
    """按调用顺序依次返回不同响应，用于模拟"先搜索、再轮询"的两段式交互。"""

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        self.urls.append(str(request.url))
        return httpx.Response(200, json=self.responses[index])


class TestXiaoBeiAsyncScrapePolling:
    """`full_content=True` 时上游走异步模式：首响应带回搜索结果、抓取仍在跑，
    真实调用几乎必然进入 `_poll_task`。这是生产主路径，必须有回归约束。"""

    def _install_sequence(self, monkeypatch: pytest.MonkeyPatch, recorder: _SequenceRecorder) -> None:
        real_client = httpx.Client

        def _factory(*_args: object, **_kwargs: object) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(recorder.handler))

        monkeypatch.setattr(agg.httpx, "Client", _factory)
        monkeypatch.setattr(agg.time, "sleep", lambda _s: None)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="XiaoBei",
            SEARCH_API_KEY="sk",
            SEARCH_TIMEOUT="60",
        )

    def test_polls_until_scrape_completes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pending = {
            "task_id": "wr_1",
            "status": "scrape_queued",
            "results": [
                {
                    "rank": 1,
                    "title": "T",
                    "url": "https://e.com",
                    "snippet": "摘要",
                    "engine": "bing",
                    "scrape": {"status": "scraping", "markdown": ""},
                }
            ],
        }
        done = {
            "task_id": "wr_1",
            "status": "complete",
            "results": [
                {
                    "rank": 1,
                    "title": "T",
                    "url": "https://e.com",
                    "snippet": "摘要",
                    "engine": "bing",
                    "scrape": {"status": "done", "markdown": "正文全文"},
                }
            ],
        }
        recorder = _SequenceRecorder([pending, pending, done])
        self._install_sequence(monkeypatch, recorder)

        result = web_search(query="q", num_results=1, full_content=True)

        assert recorder.calls >= 2, "首响应未落定时必须轮询 /tasks/{id}"
        assert any("/tasks/wr_1" in url for url in recorder.urls[1:])
        assert result["sources"][0]["snippet"] == "正文全文"

    def test_polling_stops_at_terminal_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """任务进终态就停止轮询，抓取失败时降级回搜索摘要。"""
        failed = {
            "task_id": "wr_2",
            "status": "failed",
            "results": [
                {
                    "rank": 1,
                    "title": "T",
                    "url": "https://e.com",
                    "snippet": "回落摘要",
                    "engine": "baidu",
                    "scrape": {"status": "failed", "markdown": "", "error": "timeout"},
                }
            ],
        }
        recorder = _SequenceRecorder([failed])
        self._install_sequence(monkeypatch, recorder)

        result = web_search(query="q", num_results=1, full_content=True)

        assert recorder.calls == 1, "首响应已是终态就不该再轮询"
        assert result["sources"][0]["snippet"] == "回落摘要"

    def test_poll_failure_keeps_search_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """轮询本身出错时保留已拿到的搜索结果，不能因为抓不到正文就整体失败。"""
        first = {
            "task_id": "wr_3",
            "status": "scrape_queued",
            "results": [
                {
                    "rank": 1,
                    "title": "T",
                    "url": "https://e.com",
                    "snippet": "已有摘要",
                    "engine": "google",
                    "scrape": {"status": "queued", "markdown": ""},
                }
            ],
        }
        calls = {"n": 0}
        real_client = httpx.Client

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(200, json=first)
            raise httpx.ConnectTimeout("poll boom")

        def _factory(*_args: object, **_kwargs: object) -> httpx.Client:
            return real_client(transport=httpx.MockTransport(handler))

        monkeypatch.setattr(agg.httpx, "Client", _factory)
        monkeypatch.setattr(agg.time, "sleep", lambda _s: None)
        _set(
            monkeypatch,
            SEARCH_PROVIDER="XiaoBei",
            SEARCH_API_KEY="sk",
            SEARCH_TIMEOUT="60",
        )

        result = web_search(query="q", num_results=1, full_content=True)

        assert result.get("error") is None, "轮询失败不该让整次检索失败"
        assert result["sources"][0]["snippet"] == "已有摘要"

    def test_scrape_settled_detects_pending_states(self) -> None:
        pending_payload = {
            "status": "scrape_queued",
            "results": [{"scrape": {"status": "queued"}}],
        }
        settled_payload = {
            "status": "scrape_queued",
            "results": [{"scrape": {"status": "done"}}],
        }

        assert not agg.XiaoBeiProvider._scrape_settled(pending_payload)
        assert agg.XiaoBeiProvider._scrape_settled(settled_payload)
