from __future__ import annotations

import asyncio
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider
from mcp.shared.auth import AuthorizationCodeResult, OAuthToken

from nexau.archs.main_sub.config.schema import MCPEnvSecretRef
from nexau.archs.tool.builtin.mcp_auth import (
    BearerAuth,
    EnvSecretResolver,
    MCPAuthContext,
    MCPAuthHost,
    NamespacedMemoryTokenStorage,
    build_http_auth,
    build_http_client,
    build_token_storage_namespace,
    canonicalize_mcp_url,
    redact_headers,
    redact_sensitive_data,
    redact_url,
)


def _run[T](awaitable):  # type: ignore[no-untyped-def]
    return asyncio.run(awaitable)


def test_env_secret_resolver_reads_non_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TEST_TOKEN", "resolved-token")
    resolver = EnvSecretResolver()

    value = resolver.resolve(MCPEnvSecretRef(key="MCP_TEST_TOKEN"), MCPAuthContext())

    assert value == "resolved-token"


@pytest.mark.parametrize("value", [None, ""])
def test_env_secret_resolver_rejects_missing_or_empty_value(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("MCP_EMPTY_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MCP_EMPTY_TOKEN", value)

    with pytest.raises(ValueError, match="MCP_EMPTY_TOKEN"):
        EnvSecretResolver().resolve(MCPEnvSecretRef(key="MCP_EMPTY_TOKEN"), MCPAuthContext())


def test_bearer_auth_adds_header_and_does_not_expose_token() -> None:
    auth = BearerAuth("top-secret")
    request = httpx2.Request("GET", "https://mcp.example.test/tools")

    authenticated = next(auth.auth_flow(request))

    assert authenticated.headers["Authorization"] == "Bearer top-secret"
    assert "top-secret" not in repr(auth)


def test_build_http_auth_selects_bearer_from_server_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_BEARER", "bearer-value")

    async def exercise() -> None:
        server = {
            "name": "bearer-server",
            "type": "http",
            "url": "https://mcp.example.test/mcp",
            "auth": {"type": "bearer", "token": {"source": "env", "key": "MCP_BEARER"}},
        }
        async with build_http_auth(server) as auth:
            assert isinstance(auth, BearerAuth)
            request = httpx2.Request("POST", str(server["url"]))
            authenticated = next(auth.auth_flow(request))
            assert authenticated.headers["Authorization"] == "Bearer bearer-value"

    _run(exercise())


def test_build_http_client_consumes_server_headers_and_timeout() -> None:
    async def exercise() -> None:
        server = {
            "name": "header-server",
            "type": "http",
            "url": "https://mcp.example.test/mcp",
            "headers": {"X-Tenant": "north"},
            "timeout": 7,
        }
        async with build_http_client(server) as client:
            assert isinstance(client, httpx2.AsyncClient)
            assert client.headers["X-Tenant"] == "north"
            assert client.timeout.connect == 7

    _run(exercise())


def test_authorization_code_uses_official_provider_and_seeds_registered_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_CLIENT_SECRET", "registered-secret")
    redirects: list[str] = []

    async def redirect_handler(url: str) -> None:
        redirects.append(url)

    async def callback_handler() -> AuthorizationCodeResult:
        return AuthorizationCodeResult(code="code", state="state")

    host = MCPAuthHost(
        identity="user:42",
        redirect_uri="https://app.example.test/oauth/callback",
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    context = MCPAuthContext(identity="user:42", server_name="remote", source_id="plugin:test:mcp_server:remote")
    server = {
        "name": "remote",
        "type": "http",
        "url": "https://mcp.example.test/mcp",
        "auth": {
            "type": "authorization_code",
            "client_name": "NexAU Test",
            "client_id": "registered-client",
            "client_secret": {"source": "env", "key": "MCP_CLIENT_SECRET"},
            "scopes": ["tools.write", "tools.read"],
        },
    }

    async def exercise() -> None:
        async with build_http_auth(server, auth_host=host, auth_context=context) as auth:
            assert type(auth) is OAuthClientProvider
            assert auth.context.client_metadata.client_name == "NexAU Test"
            assert auth.context.client_metadata.scope == "tools.read tools.write"
            assert auth.context.client_metadata.redirect_uris is not None
            assert str(auth.context.client_metadata.redirect_uris[0]) == "https://app.example.test/oauth/callback"
            client_info = await auth.context.storage.get_client_info()
            assert client_info is not None
            assert client_info.client_id == "registered-client"
            assert client_info.client_secret == "registered-secret"

    _run(exercise())
    assert redirects == []


def test_authorization_code_reuses_tokens_with_same_host() -> None:
    async def redirect_handler(url: str) -> None:
        del url

    async def callback_handler() -> AuthorizationCodeResult:
        return AuthorizationCodeResult(code="unused")

    host = MCPAuthHost(
        identity="user:reuse",
        redirect_uri="https://app.example.test/oauth/callback",
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    server = {
        "name": "remote",
        "url": "https://mcp.example.test/mcp",
        "auth": {"type": "authorization_code", "client_name": "NexAU", "scopes": ["tools.read"]},
    }

    async def exercise() -> None:
        async with build_http_auth(server, auth_host=host) as first:
            assert isinstance(first, OAuthClientProvider)
            await first.context.storage.set_tokens(OAuthToken(access_token="reused-token", refresh_token="refresh"))
        async with build_http_auth(server, auth_host=host) as second:
            assert isinstance(second, OAuthClientProvider)
            stored = await second.context.storage.get_tokens()
            assert stored is not None
            assert stored.access_token == "reused-token"

    _run(exercise())


def test_client_credentials_uses_official_noninteractive_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_MACHINE_SECRET", "machine-secret")
    interactions: list[str] = []

    async def redirect_handler(url: str) -> None:
        interactions.append(url)

    async def callback_handler() -> AuthorizationCodeResult:
        interactions.append("callback")
        return AuthorizationCodeResult(code="unexpected")

    host = MCPAuthHost(
        identity="service:worker",
        redirect_uri="https://app.example.test/oauth/callback",
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    server = {
        "name": "machine",
        "type": "http",
        "url": "https://mcp.example.test/mcp",
        "auth": {
            "type": "client_credentials",
            "client_id": "worker-client",
            "client_secret": {"source": "env", "key": "MCP_MACHINE_SECRET"},
            "scopes": ["tools.read"],
        },
    }

    async def exercise() -> None:
        async with build_http_auth(server, auth_host=host) as auth:
            assert isinstance(auth, ClientCredentialsOAuthProvider)
            assert auth.context.redirect_handler is None
            assert auth.context.callback_handler is None
            assert auth.context.client_metadata.scope == "tools.read"

    _run(exercise())
    assert interactions == []


def test_token_storage_is_isolated_by_identity_server_client_url_and_scopes() -> None:
    first_namespace = build_token_storage_namespace(
        context=MCPAuthContext(identity="user:1", server_name="remote", source_id="source:remote"),
        server_url="https://MCP.example.test:443/mcp#fragment",
        client_id="client",
        scopes=["b", "a", "a"],
    )
    equivalent_namespace = build_token_storage_namespace(
        context=MCPAuthContext(identity="user:1", server_name="remote", source_id="source:remote"),
        server_url="https://mcp.example.test/mcp",
        client_id="client",
        scopes=["a", "b"],
    )
    other_namespace = build_token_storage_namespace(
        context=MCPAuthContext(identity="user:2", server_name="remote", source_id="source:remote"),
        server_url="https://MCP.example.test:443/mcp#fragment",
        client_id="client",
        scopes=["b", "a", "a"],
    )
    host = MCPAuthHost()
    first = host.get_token_storage(first_namespace)
    equivalent = host.get_token_storage(equivalent_namespace)
    other = host.get_token_storage(other_namespace)

    async def exercise() -> None:
        await first.set_tokens(OAuthToken(access_token="token-one"))
        assert (await equivalent.get_tokens()).access_token == "token-one"  # type: ignore[union-attr]
        assert await other.get_tokens() is None

    _run(exercise())


def test_token_storage_returns_defensive_copies() -> None:
    namespace = build_token_storage_namespace(
        context=MCPAuthContext(identity="user", server_name="server"),
        server_url="https://mcp.example.test",
        client_id=None,
        scopes=[],
    )
    storage = NamespacedMemoryTokenStorage(namespace)

    async def exercise() -> None:
        original = OAuthToken(access_token="one")
        await storage.set_tokens(original)
        loaded = await storage.get_tokens()
        assert loaded is not None
        loaded.access_token = "mutated"
        reloaded = await storage.get_tokens()
        assert reloaded is not None
        assert reloaded.access_token == "one"

    _run(exercise())


def test_default_cli_host_uses_isolated_loopback_callback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = MCPAuthHost(open_browser=False, authorization_timeout=2)

    class FakeSocket:
        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 43123)

    class FakeServer:
        sockets = [FakeSocket()]

        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    async def fake_start_server(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return FakeServer()

    monkeypatch.setattr(asyncio, "start_server", fake_start_server)

    async def exercise() -> None:
        async with host.authorization_code_session(MCPAuthContext()) as session:
            parsed = urlsplit(session.redirect_uri)
            assert parsed.scheme == "http"
            assert parsed.hostname == "127.0.0.1"
            assert parsed.port == 43123
            await session.redirect_handler("https://auth.example.test/authorize?state=expected")

    _run(exercise())
    assert "https://auth.example.test/authorize" in capsys.readouterr().out


def test_custom_storage_factory_receives_complete_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_MACHINE_SECRET", "machine-secret")
    namespaces = []

    def factory(namespace):  # type: ignore[no-untyped-def]
        namespaces.append(namespace)
        return NamespacedMemoryTokenStorage(namespace)

    host = MCPAuthHost(identity="tenant:user", token_storage_factory=factory)
    server = {
        "name": "machine",
        "source_id": "plugin:test:mcp_server:machine",
        "url": "https://mcp.example.test/mcp",
        "auth": {
            "type": "client_credentials",
            "client_id": "client",
            "client_secret": {"source": "env", "key": "MCP_MACHINE_SECRET"},
            "scopes": ["write", "read"],
        },
    }

    async def exercise() -> None:
        async with build_http_auth(server, auth_host=host):
            pass

    _run(exercise())
    assert len(namespaces) == 1
    namespace = namespaces[0]
    assert namespace.identity == "tenant:user"
    assert namespace.server_name == "machine"
    assert namespace.source_id == "plugin:test:mcp_server:machine"
    assert namespace.canonical_url == "https://mcp.example.test/mcp"
    assert namespace.client_id == "client"
    assert namespace.scopes == ("read", "write")


def test_default_cli_auth_host_reuses_loopback_redirect_uri() -> None:
    host = MCPAuthHost(open_browser=False)
    context = MCPAuthContext(identity="user", server_name="server")

    async def exercise() -> tuple[str, str]:
        async with host.authorization_code_session(context) as first:
            first_uri = first.redirect_uri
        async with host.authorization_code_session(context) as second:
            second_uri = second.redirect_uri
        return first_uri, second_uri

    first_uri, second_uri = asyncio.run(exercise())

    assert first_uri == second_uri
    assert first_uri.startswith("http://127.0.0.1:")


def test_canonicalize_mcp_url_normalizes_origin_and_removes_fragment() -> None:
    assert canonicalize_mcp_url("HTTPS://Example.COM:443/path?tenant=1#secret") == "https://example.com/path?tenant=1"
    assert canonicalize_mcp_url("http://[::1]:80") == "http://[::1]/"


def test_redact_headers_is_case_insensitive_and_non_mutating() -> None:
    headers = {"authorization": "Bearer secret", "Proxy-Authorization": "Basic secret", "X-Tenant": "north"}

    redacted = redact_headers(headers)

    assert redacted == {"authorization": "[REDACTED]", "Proxy-Authorization": "[REDACTED]", "X-Tenant": "north"}
    assert headers["authorization"] == "Bearer secret"


def test_redact_url_hides_query_password_and_fragment() -> None:
    redacted = redact_url("https://user:password@example.test/mcp?key=secret&tenant=north&access_token=token#fragment")
    parsed = urlsplit(redacted)
    query = parse_qs(parsed.query)

    assert "password" not in redacted
    assert "secret" not in redacted
    assert "=token" not in redacted
    assert "fragment" not in redacted
    assert query == {"key": ["[REDACTED]"], "tenant": ["north"], "access_token": ["[REDACTED]"]}


def test_redact_sensitive_data_handles_nested_fields_and_explicit_values() -> None:
    payload = {
        "client_secret": "one",
        "nested": {"refresh-token": "two", "message": "request failed with resolved-secret"},
        "url": "https://mcp.example.test?api_key=three",
    }

    redacted = redact_sensitive_data(payload, secrets=("resolved-secret",))

    assert redacted["client_secret"] == "[REDACTED]"
    assert redacted["nested"]["refresh-token"] == "[REDACTED]"
    assert redacted["nested"]["message"] == "request failed with [REDACTED]"
    assert "three" not in redacted["url"]


def test_redact_sensitive_data_sanitizes_credentials_and_embedded_urls_in_errors() -> None:
    message = (
        "request to https://user:password@example.test/mcp?key=query-secret failed; "
        "Authorization: Bearer header-secret; access_token=body-secret"
    )

    redacted = redact_sensitive_data(message)

    assert isinstance(redacted, str)
    assert "password" not in redacted
    assert "query-secret" not in redacted
    assert "header-secret" not in redacted
    assert "body-secret" not in redacted
    assert redacted.count("[REDACTED]") >= 2
    assert "%5BREDACTED%5D" in redacted


def test_host_rejects_partial_custom_callback_configuration() -> None:
    async def redirect_handler(url: str) -> None:
        del url

    with pytest.raises(ValueError, match="provided together"):
        MCPAuthHost(redirect_uri="https://app.example.test/callback", redirect_handler=redirect_handler)
