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

"""`web_read` 两条取内容路径都要经出网代理。

私有化部署里沙箱只有受控网关一条出口，漏接线的表现不是报错，而是**静默超时**——
看起来像"这个网页读不了"。所以这里不测 `resolve_proxy()` 解析得对不对
（那在 test_aggregated_search.py 里），只钉死配置真的传到了 `httpx.Client`。

用例都走 MockTransport，不打真实上游，CI 不需要任何凭据。
"""

from typing import Any

import httpx
import pytest

from nexau.archs.tool.builtin.web_tools import proxy as proxy_mod
from nexau.archs.tool.builtin.web_tools import web_tool

_PARSER_ENVS = ("BP_HTML_PARSER_URL", "BP_HTML_PARSER_API_KEY", "BP_HTML_PARSER_SECRET")
_PROXY_ENVS = (proxy_mod.PROXY_URL_ENV, proxy_mod.PROXY_AUTH_ENV)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """开发机上真配了代理或解析服务时，断言会拿到非预期值。"""
    for key in (*_PARSER_ENVS, *_PROXY_ENVS):
        monkeypatch.delenv(key, raising=False)
    # 模块级单例会跨用例带住上一轮读到的配置
    web_tool._html_parser = None


def _capture(monkeypatch: pytest.MonkeyPatch, payload: object) -> dict[str, Any]:
    """替换 httpx.Client，把构造 kwargs 抓下来，并让请求走 MockTransport。"""
    seen: dict[str, Any] = {}
    real_client = httpx.Client

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    def _factory(*args: object, **kwargs: Any) -> httpx.Client:
        seen.update(kwargs)
        kwargs.pop("transport", None)
        kwargs.pop("proxy", None)
        return real_client(transport=httpx.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(web_tool.httpx, "Client", _factory)
    return seen


def _configure_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BP_HTML_PARSER_URL", "https://parser.internal/url2md")
    monkeypatch.setenv("BP_HTML_PARSER_API_KEY", "k")
    monkeypatch.setenv("BP_HTML_PARSER_SECRET", "s")


def _set_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(proxy_mod.PROXY_URL_ENV, "http://gw:8444")
    monkeypatch.setenv(proxy_mod.PROXY_AUTH_ENV, "Bearer tk-123")


def _assert_configured_proxy(seen: dict[str, Any]) -> None:
    proxy = seen.get("proxy")
    assert isinstance(proxy, httpx.Proxy), "proxy 没有传给 httpx.Client，代理配置形同虚设"
    assert str(proxy.url) == "http://gw:8444"
    assert proxy.headers["Proxy-Authorization"] == "Bearer tk-123"


class TestHtmlParserProxy:
    """HTML 解析服务这一跳。"""

    def test_configured_proxy_reaches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _configure_parser(monkeypatch)
        _set_proxy(monkeypatch)
        seen = _capture(monkeypatch, {"content": "# hi"})

        result = web_tool.web_read("https://example.com/a")

        assert result["method"] == "html_parser", "应当命中解析服务而不是回退直抓"
        _assert_configured_proxy(seen)

    def test_unconfigured_passes_none_and_keeps_trust_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """不配代理时必须是 None，httpx 才会继续按 trust_env 读标准变量。"""
        _configure_parser(monkeypatch)
        seen = _capture(monkeypatch, {"content": "# hi"})

        web_tool.web_read("https://example.com/a")

        assert "proxy" in seen, "应显式传 proxy 参数"
        assert seen["proxy"] is None, "未配置时必须是 None，否则会覆盖 trust_env 行为"


class TestDirectFetchProxy:
    """解析服务未配置时的回退直抓这一跳。"""

    def test_configured_proxy_reaches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_proxy(monkeypatch)
        seen = _capture(monkeypatch, {"ok": True})

        result = web_tool.web_read("https://example.com/a")

        assert result["method"] == "direct_http", "解析服务没配，应当走回退路径"
        _assert_configured_proxy(seen)

    def test_unconfigured_passes_none_and_keeps_trust_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = _capture(monkeypatch, {"ok": True})

        web_tool.web_read("https://example.com/a")

        assert "proxy" in seen, "应显式传 proxy 参数"
        assert seen["proxy"] is None, "未配置时必须是 None，否则会覆盖 trust_env 行为"

    def test_timeout_still_reaches_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """加 proxy 不能把原有的 timeout 挤掉。"""
        seen = _capture(monkeypatch, {"ok": True})

        web_tool.web_read("https://example.com/a", timeout=7)

        assert seen.get("timeout") == 7


class TestForcedTunnelGate:
    """强制隧道只接管「配了代理 + 明文 parser」这一种组合。

    这条新路径的风险全在**误接管**：一旦它把本来好好走 httpx 的部署抢过去，
    影响的是所有配了解析服务的环境。所以这里逐个钉死不该接管的组合。
    """

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """记录隧道路径是否被调用，并让它返回一个成功响应。"""
        calls: dict[str, Any] = {"tunneled": False}

        def _post_json(target_url, proxy, *, payload, headers, timeout):  # noqa: ANN001, ANN202
            calls.update(
                tunneled=True,
                target_url=target_url,
                proxy_url=str(proxy.url),
                payload=payload,
                headers=headers,
                timeout=timeout,
            )
            return 200, b'{"content": "# tunneled"}'

        monkeypatch.setattr(web_tool.tunnel, "post_json", _post_json)
        return calls

    def test_http_parser_with_proxy_is_tunneled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BP_HTML_PARSER_URL", "http://parser.internal:10010/url2md")
        monkeypatch.setenv("BP_HTML_PARSER_API_KEY", "k")
        monkeypatch.setenv("BP_HTML_PARSER_SECRET", "s")
        _set_proxy(monkeypatch)
        calls = self._spy(monkeypatch)
        seen = _capture(monkeypatch, {"content": "# via httpx"})

        result = web_tool.web_read("https://example.com/a")

        assert calls["tunneled"] is True, "明文 parser + 代理必须走隧道，否则线上是 407"
        assert result["content"] == "# tunneled"
        assert seen == {}, "走了隧道就不该再构造 httpx.Client"
        # 三个签名头必须原样进隧道内那条业务请求
        assert set(calls["headers"]) == {"X-API-KEY", "X-TIMESTAMP", "X-SIGNATURE"}
        assert calls["payload"] == {"url": "https://example.com/a"}

    def test_https_parser_stays_on_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """https 目标 httpx 自己就会发 CONNECT，不需要接管。"""
        _configure_parser(monkeypatch)  # https://parser.internal/url2md
        _set_proxy(monkeypatch)
        calls = self._spy(monkeypatch)
        seen = _capture(monkeypatch, {"content": "# via httpx"})

        result = web_tool.web_read("https://example.com/a")

        assert calls["tunneled"] is False
        assert result["content"] == "# via httpx"
        _assert_configured_proxy(seen)

    def test_http_parser_without_proxy_stays_on_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """没配代理就是直连，不存在 CONNECT 这回事——绝大多数部署走这里。"""
        monkeypatch.setenv("BP_HTML_PARSER_URL", "http://parser.internal:10010/url2md")
        monkeypatch.setenv("BP_HTML_PARSER_API_KEY", "k")
        monkeypatch.setenv("BP_HTML_PARSER_SECRET", "s")
        calls = self._spy(monkeypatch)
        seen = _capture(monkeypatch, {"content": "# via httpx"})

        result = web_tool.web_read("https://example.com/a")

        assert calls["tunneled"] is False
        assert result["content"] == "# via httpx"
        assert seen["proxy"] is None

    def test_https_proxy_is_not_taken_over(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """代理本身要 TLS 时标准库那条走不通，明确不接管而不是崩掉。"""
        monkeypatch.setenv("BP_HTML_PARSER_URL", "http://parser.internal:10010/url2md")
        monkeypatch.setenv("BP_HTML_PARSER_API_KEY", "k")
        monkeypatch.setenv("BP_HTML_PARSER_SECRET", "s")
        monkeypatch.setenv(proxy_mod.PROXY_URL_ENV, "https://gw:8444")
        calls = self._spy(monkeypatch)
        _capture(monkeypatch, {"content": "# via httpx"})

        web_tool.web_read("https://example.com/a")

        assert calls["tunneled"] is False

    def test_tunnel_failure_degrades_to_direct_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """隧道抛异常时沿用既有降级语义，不把异常抛给调用方。"""
        monkeypatch.setenv("BP_HTML_PARSER_URL", "http://parser.internal:10010/url2md")
        monkeypatch.setenv("BP_HTML_PARSER_API_KEY", "k")
        monkeypatch.setenv("BP_HTML_PARSER_SECRET", "s")
        _set_proxy(monkeypatch)

        def _boom(*_args: object, **_kwargs: object) -> tuple[int, bytes]:
            raise OSError("tunnel refused")

        monkeypatch.setattr(web_tool.tunnel, "post_json", _boom)
        _capture(monkeypatch, {"ok": True})

        result = web_tool.web_read("https://example.com/a")

        assert result["method"] == "direct_http", "解析服务失败应回退直抓，与改动前一致"


class TestTunnelWireFormat:
    """隧道那一跳发出去的字节。

    上面那组用 mock 证明"选路对"，这组用真 socket 证明"发出去的东西对"——
    两者缺一：选路对但 CONNECT 写错，线上照样 407。
    """

    def test_connect_carries_proxy_auth_then_plaintext_request(self) -> None:
        import json as _json
        import socket
        import threading

        seen: dict[str, Any] = {}
        origin_payload = _json.dumps({"content": "# parsed"}).encode()

        def _origin(sock: socket.socket) -> None:
            f = sock.makefile("rb")
            seen["request_line"] = f.readline().decode().strip()
            hdrs = {}
            while True:
                raw = f.readline().decode().strip()
                if not raw:
                    break
                k, _, v = raw.partition(":")
                hdrs[k.strip()] = v.strip()
            seen["request_headers"] = hdrs
            f.read(int(hdrs.get("Content-Length", 0)))
            sock.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(origin_payload)).encode()
                + b"\r\n\r\n"
                + origin_payload
            )

        def _proxy(sock: socket.socket) -> None:
            f = sock.makefile("rb")
            seen["connect_line"] = f.readline().decode().strip()
            while True:
                raw = f.readline().decode().strip()
                if not raw:
                    break
                if raw.lower().startswith("proxy-authorization:"):
                    seen["connect_auth"] = raw.partition(":")[2].strip()
            sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _origin(sock)  # 隧道建成后由同一条连接继续扮演目标服务

        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        server = threading.Thread(target=lambda: _proxy(listener.accept()[0]), daemon=True)
        server.start()

        from nexau.archs.tool.builtin.web_tools import tunnel as tunnel_mod

        status, body = tunnel_mod.post_json(
            "http://parser.internal:10010/url2md",
            httpx.Proxy(f"http://127.0.0.1:{port}", headers={"Proxy-Authorization": "Bearer tk"}),
            payload={"url": "https://example.com"},
            headers={"X-API-KEY": "k"},
            timeout=5,
        )
        server.join(timeout=5)
        listener.close()

        assert status == 200
        assert _json.loads(body)["content"] == "# parsed"
        assert seen["connect_line"] == "CONNECT parser.internal:10010 HTTP/1.1"
        assert seen["connect_auth"] == "Bearer tk", "凭据必须在 CONNECT 上，不是业务请求上"
        assert seen["request_line"] == "POST /url2md HTTP/1.1", "隧道内应是 origin-form 明文请求"
        assert seen["request_headers"]["X-API-KEY"] == "k"
        assert "Proxy-Authorization" not in seen["request_headers"], "业务请求不该重复带代理凭据"
