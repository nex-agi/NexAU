"""Authentication adapters for the official MCP Python SDK.

RFC-0029 keeps OAuth protocol behavior in the SDK.  This module only resolves
host-owned secrets, scopes token storage, supplies interactive callbacks, and
constructs SDK authentication providers and HTTP clients.
"""

from __future__ import annotations

import asyncio
import os
import re
import webbrowser
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, Protocol, cast
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx2
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl, BaseModel

from nexau.archs.main_sub.config.schema import MCPEnvSecretRef

_REDACTED = "[REDACTED]"
_SENSITIVE_HEADER_NAMES = {"authorization", "proxy-authorization"}
_SENSITIVE_FIELD_NAMES = {
    "access_token",
    "authorization",
    "client_secret",
    "id_token",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "token",
}
_SENSITIVE_QUERY_NAMES = _SENSITIVE_FIELD_NAMES | {"api_key", "apikey", "key"}
_URL_IN_TEXT_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_BEARER_IN_TEXT_PATTERN = re.compile(r"\bBearer\s+[^\s,;\"']+", re.IGNORECASE)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:access[_-]?token|refresh[_-]?token|client[_-]?secret|api[_-]?key|token)\b\s*[:=]\s*)([^\s,;}&]+)"
)


@dataclass(frozen=True, slots=True)
class MCPAuthContext:
    """Host identity used to isolate credentials belonging to an MCP server."""

    identity: str = "local"
    server_name: str = "unknown"
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class MCPTokenStorageNamespace:
    """Complete isolation key for OAuth tokens and registered client data."""

    identity: str
    server_name: str
    source_id: str
    canonical_url: str = field(repr=False)
    client_id: str
    scopes: tuple[str, ...]


class SecretResolver(Protocol):
    """Resolve a typed secret reference within a host identity boundary."""

    def resolve(self, reference: MCPEnvSecretRef, context: MCPAuthContext) -> str: ...


class EnvSecretResolver:
    """Resolve RFC-0029 ``source: env`` secret references."""

    def resolve(self, reference: MCPEnvSecretRef, context: MCPAuthContext) -> str:
        del context
        value = os.environ.get(reference.key)
        if value is None or not value:
            raise ValueError(f"MCP auth environment variable '{reference.key}' is not set or is empty")
        return value


@dataclass(frozen=True, slots=True)
class MCPAuthorizationCodeSession:
    """Host callbacks consumed by the official authorization-code provider."""

    redirect_uri: str
    redirect_handler: Callable[[str], Awaitable[None]]
    callback_handler: Callable[[], Awaitable[AuthorizationCodeResult]]


class BearerAuth(httpx2.Auth):
    """Minimal static bearer auth; it never starts an OAuth flow."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("Bearer token must not be empty")
        self._token = token

    def auth_flow(self, request: httpx2.Request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request

    def __repr__(self) -> str:
        return "BearerAuth(token='[REDACTED]')"


@dataclass(slots=True)
class _MemoryStorageState:
    tokens: dict[MCPTokenStorageNamespace, OAuthToken] = field(default_factory=lambda: cast(dict[MCPTokenStorageNamespace, OAuthToken], {}))
    client_info: dict[MCPTokenStorageNamespace, OAuthClientInformationFull] = field(
        default_factory=lambda: cast(dict[MCPTokenStorageNamespace, OAuthClientInformationFull], {})
    )
    lock: RLock = field(default_factory=RLock)


class NamespacedMemoryTokenStorage(TokenStorage):
    """In-memory SDK TokenStorage isolated by an explicit namespace.

    This storage is intended for tests, CLI sessions, and short-lived processes.
    Production hosts should inject encrypted persistent storage through
    :class:`MCPAuthHost`.
    """

    def __init__(self, namespace: MCPTokenStorageNamespace, state: _MemoryStorageState | None = None) -> None:
        self.namespace = namespace
        self._state = state or _MemoryStorageState()

    async def get_tokens(self) -> OAuthToken | None:
        with self._state.lock:
            tokens = self._state.tokens.get(self.namespace)
            return tokens.model_copy(deep=True) if tokens is not None else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        with self._state.lock:
            self._state.tokens[self.namespace] = tokens.model_copy(deep=True)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        with self._state.lock:
            client_info = self._state.client_info.get(self.namespace)
            return client_info.model_copy(deep=True) if client_info is not None else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        with self._state.lock:
            self._state.client_info[self.namespace] = client_info.model_copy(deep=True)


class _RedirectAwareTokenStorage(TokenStorage):
    """Ignore dynamic registrations tied to a different loopback redirect URI."""

    def __init__(self, storage: TokenStorage, redirect_uri: str) -> None:
        self._storage = storage
        self._redirect_uri = redirect_uri

    async def get_tokens(self) -> OAuthToken | None:
        return await self._storage.get_tokens()

    async def set_tokens(self, tokens: OAuthToken) -> None:
        await self._storage.set_tokens(tokens)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        client_info = await self._storage.get_client_info()
        if client_info is None or client_info.redirect_uris is None:
            return client_info
        redirect_uris = {str(uri) for uri in client_info.redirect_uris}
        return client_info if self._redirect_uri in redirect_uris else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        await self._storage.set_client_info(client_info)


class _LoopbackAuthorizationServer:
    """One-flow loopback receiver used by the default CLI auth host."""

    def __init__(self, *, timeout: float, open_browser: bool, port: int = 0) -> None:
        self._timeout = timeout
        self._open_browser = open_browser
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[AuthorizationCodeResult] | None = None

    async def __aenter__(self) -> MCPAuthorizationCodeSession:
        loop = asyncio.get_running_loop()
        self._result = loop.create_future()
        self._server = await asyncio.start_server(self._handle_request, "127.0.0.1", self._port)
        sockets: list[Any] = list(self._server.sockets or [])
        if not sockets:
            raise RuntimeError("OAuth loopback callback server did not bind a socket")
        port = cast(tuple[str, int], sockets[0].getsockname())[:2][1]
        return MCPAuthorizationCodeSession(
            redirect_uri=f"http://127.0.0.1:{port}/callback",
            redirect_handler=self._redirect,
            callback_handler=self._callback,
        )

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._result is not None and not self._result.done():
            self._result.cancel()

    async def _redirect(self, authorization_url: str) -> None:
        opened = False
        if self._open_browser:
            opened = await asyncio.to_thread(webbrowser.open, authorization_url)
        if not opened:
            print(f"Open this URL to authorize the MCP server:\n{authorization_url}")

    async def _callback(self) -> AuthorizationCodeResult:
        if self._result is None:
            raise RuntimeError("OAuth loopback callback session is not active")
        return await asyncio.wait_for(asyncio.shield(self._result), timeout=self._timeout)

    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status = "400 Bad Request"
        body = b"OAuth callback was invalid. You may close this window."
        try:
            request_head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
            first_line = request_head.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
            method, target, _version = first_line.split(" ", 2)
            parsed = urlsplit(target)
            params = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if method != "GET" or parsed.path != "/callback":
                raise ValueError("unexpected callback request")
            if "error" in params:
                description = params.get("error_description") or params["error"]
                raise ValueError(f"OAuth authorization failed: {description}")
            code = params.get("code")
            if not code:
                raise ValueError("OAuth callback did not include an authorization code")
            result = AuthorizationCodeResult(code=code, state=params.get("state"), iss=params.get("iss"))
            if self._result is not None and not self._result.done():
                self._result.set_result(result)
            status = "200 OK"
            body = b"MCP authorization completed. You may close this window."
        except Exception as error:
            if self._result is not None and not self._result.done():
                self._result.set_exception(error)
        finally:
            response = (
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            ).encode("ascii") + body
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()


TokenStorageFactory = Callable[[MCPTokenStorageNamespace], TokenStorage]


class MCPAuthHost:
    """Injectable host boundary for MCP secrets, storage, and user interaction.

    With no callback arguments this is a safe local CLI host: it binds an
    ephemeral loopback listener, opens the system browser when possible, and
    otherwise prints the authorization URL.  Web/server hosts should inject a
    stable HTTPS redirect URI plus their own redirect and callback handlers.
    """

    def __init__(
        self,
        *,
        identity: str = "local",
        secret_resolver: SecretResolver | None = None,
        token_storage_factory: TokenStorageFactory | None = None,
        redirect_uri: str | None = None,
        redirect_handler: Callable[[str], Awaitable[None]] | None = None,
        callback_handler: Callable[[], Awaitable[AuthorizationCodeResult]] | None = None,
        authorization_timeout: float = 300.0,
        open_browser: bool = True,
    ) -> None:
        custom_callbacks = (redirect_handler is not None, callback_handler is not None, redirect_uri is not None)
        if any(custom_callbacks) and not all(custom_callbacks):
            raise ValueError("redirect_uri, redirect_handler, and callback_handler must be provided together")
        if authorization_timeout <= 0:
            raise ValueError("authorization_timeout must be positive")
        self.identity = identity
        self.secret_resolver = secret_resolver or EnvSecretResolver()
        self._memory_state = _MemoryStorageState()
        self._token_storage_factory = token_storage_factory
        self._redirect_uri = redirect_uri
        self._redirect_handler = redirect_handler
        self._callback_handler = callback_handler
        self._authorization_timeout = authorization_timeout
        self._open_browser = open_browser
        # Keep the default CLI redirect URI stable for this host so dynamic
        # client registration and refresh-token state remain reusable across
        # bootstrap discovery and later Agent run scopes.
        self._loopback_port = 0

    def resolve_secret(self, reference: MCPEnvSecretRef, context: MCPAuthContext) -> str:
        return self.secret_resolver.resolve(reference, context)

    def get_token_storage(self, namespace: MCPTokenStorageNamespace) -> TokenStorage:
        if self._token_storage_factory is not None:
            return self._token_storage_factory(namespace)
        return NamespacedMemoryTokenStorage(namespace, self._memory_state)

    @asynccontextmanager
    async def authorization_code_session(
        self,
        context: MCPAuthContext,
    ) -> AsyncIterator[MCPAuthorizationCodeSession]:
        del context
        if self._redirect_uri is not None:
            assert self._redirect_handler is not None
            assert self._callback_handler is not None
            yield MCPAuthorizationCodeSession(
                redirect_uri=self._redirect_uri,
                redirect_handler=self._redirect_handler,
                callback_handler=self._callback_handler,
            )
            return

        loopback = _LoopbackAuthorizationServer(
            timeout=self._authorization_timeout,
            open_browser=self._open_browser,
            port=self._loopback_port,
        )
        async with loopback as session:
            self._loopback_port = urlsplit(session.redirect_uri).port or 0
            yield session


def canonicalize_mcp_url(url: str) -> str:
    """Canonicalize an MCP URL for credential-storage isolation."""

    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def build_token_storage_namespace(
    *,
    context: MCPAuthContext,
    server_url: str,
    client_id: str | None,
    scopes: list[str] | tuple[str, ...],
) -> MCPTokenStorageNamespace:
    """Build a stable, order-insensitive namespace for SDK TokenStorage."""

    normalized_scopes = tuple(sorted({scope.strip() for scope in scopes if scope.strip()}))
    return MCPTokenStorageNamespace(
        identity=context.identity,
        server_name=context.server_name,
        source_id=context.source_id or context.server_name,
        canonical_url=canonicalize_mcp_url(server_url),
        client_id=client_id or "<dynamic>",
        scopes=normalized_scopes,
    )


def redact_headers(headers: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of HTTP headers with authentication values removed."""

    if headers is None:
        return {}
    return {key: _REDACTED if key.lower() in _SENSITIVE_HEADER_NAMES else value for key, value in headers.items()}


def redact_url(url: str) -> str:
    """Redact credentials, sensitive query parameters, and URL fragments."""

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        if parsed.username is not None:
            userinfo = parsed.username
            if parsed.password is not None:
                userinfo = f"{userinfo}:{quote(_REDACTED, safe='')}"
            host = f"{userinfo}@{host}"
        query = urlencode(
            [
                (key, _REDACTED if _normalize_sensitive_name(key) in _SENSITIVE_QUERY_NAMES else value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        fragment = _REDACTED if parsed.fragment else ""
        return urlunsplit((parsed.scheme, host, parsed.path, query, fragment))
    except ValueError:
        return "<invalid-url>"


def redact_sensitive_data(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    """Recursively redact known secret fields and explicit resolved values."""

    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        return {
            key: _REDACTED
            if _normalize_sensitive_name(str(key)) in _SENSITIVE_FIELD_NAMES
            else redact_sensitive_data(item, secrets=secrets)
            for key, item in mapping_value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item, secrets=secrets) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item, secrets=secrets) for item in cast(tuple[object, ...], value))
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, _REDACTED)
        redacted = _BEARER_IN_TEXT_PATTERN.sub(f"Bearer {_REDACTED}", redacted)
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}{_REDACTED}", redacted)
        redacted = _URL_IN_TEXT_PATTERN.sub(lambda match: _redact_embedded_url(match.group(0)), redacted)
        return redacted
    return value


def _redact_embedded_url(value: str) -> str:
    """Redact a URL embedded in prose while retaining trailing punctuation."""
    url = value.rstrip(".,;:)]}")
    suffix = value[len(url) :]
    return f"{redact_url(url)}{suffix}"


def _normalize_sensitive_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return cast(Mapping[str, Any], value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    raise TypeError(f"MCP server/auth configuration must be a mapping, got {type(value).__name__}")


def _secret_ref(value: object) -> MCPEnvSecretRef:
    if isinstance(value, MCPEnvSecretRef):
        return value
    return MCPEnvSecretRef.model_validate(value)


def _context_for_server(server: Mapping[str, Any], host: MCPAuthHost, context: MCPAuthContext | None) -> MCPAuthContext:
    if context is not None:
        return context
    return MCPAuthContext(
        identity=host.identity,
        server_name=str(server.get("name") or "unknown"),
        source_id=str(server["source_id"]) if server.get("source_id") is not None else None,
    )


@asynccontextmanager
async def build_http_auth(
    server_config: object,
    *,
    auth_host: MCPAuthHost | None = None,
    auth_context: MCPAuthContext | None = None,
) -> AsyncIterator[httpx2.Auth | None]:
    """Build the configured HTTP auth strategy using official SDK providers.

    Runtime owners should create one :class:`MCPAuthHost` per Agent/identity and
    reuse it across run scopes.  Omitting ``auth_host`` intentionally creates a
    short-lived local host, so its in-memory tokens do not survive this call.
    """

    server = _mapping(server_config)
    auth_value = server.get("auth")
    if auth_value is None:
        yield None
        return

    auth = _mapping(auth_value)
    auth_type = auth.get("type")
    server_url = str(server.get("url") or "")
    if not server_url:
        raise ValueError("Remote MCP authentication requires a server URL")
    host = auth_host or MCPAuthHost()
    context = _context_for_server(server, host, auth_context)

    if auth_type == "bearer":
        token = host.resolve_secret(_secret_ref(auth.get("token")), context)
        yield BearerAuth(token)
        return

    scopes_value: object = auth.get("scopes") or []
    if not isinstance(scopes_value, (list, tuple)):
        raise ValueError("MCP OAuth scopes must be a list of strings")
    scope_items = cast(list[object] | tuple[object, ...], scopes_value)
    if not all(isinstance(scope, str) for scope in scope_items):
        raise ValueError("MCP OAuth scopes must be a list of strings")
    scopes = [cast(str, scope) for scope in scope_items]
    client_id_value = auth.get("client_id")
    client_id = str(client_id_value) if client_id_value is not None else None
    namespace = build_token_storage_namespace(
        context=context,
        server_url=server_url,
        client_id=client_id,
        scopes=scopes,
    )
    storage = host.get_token_storage(namespace)
    scope = " ".join(namespace.scopes) or None

    if auth_type == "client_credentials":
        if not client_id:
            raise ValueError("client_credentials auth requires client_id")
        secret = host.resolve_secret(_secret_ref(auth.get("client_secret")), context)
        yield ClientCredentialsOAuthProvider(
            server_url=server_url,
            storage=storage,
            client_id=client_id,
            client_secret=secret,
            scope=scope,
        )
        return

    if auth_type != "authorization_code":
        raise ValueError(f"Unsupported MCP auth type: {auth_type!r}")

    client_secret_ref = auth.get("client_secret")
    if client_secret_ref is not None and not client_id:
        raise ValueError("authorization_code client_secret requires client_id")

    async with host.authorization_code_session(context) as authorization:
        redirect_storage: TokenStorage = _RedirectAwareTokenStorage(storage, authorization.redirect_uri)
        client_secret = host.resolve_secret(_secret_ref(client_secret_ref), context) if client_secret_ref is not None else None
        token_endpoint_auth_method: Literal["client_secret_basic", "none"] = "client_secret_basic" if client_secret is not None else "none"
        client_metadata = OAuthClientMetadata(
            client_name=str(auth.get("client_name") or "NexAU"),
            redirect_uris=[AnyUrl(authorization.redirect_uri)],
            scope=scope,
            token_endpoint_auth_method=token_endpoint_auth_method,
        )
        if client_id is not None:
            stored_client = await redirect_storage.get_client_info()
            stored_redirects: set[str] = set()
            if stored_client is not None and stored_client.redirect_uris is not None:
                stored_redirects = {str(uri) for uri in stored_client.redirect_uris}
            registration_changed = (
                stored_client is None
                or stored_client.client_id != client_id
                or stored_client.client_secret != client_secret
                or stored_client.token_endpoint_auth_method != token_endpoint_auth_method
                or authorization.redirect_uri not in stored_redirects
                or stored_client.scope != scope
            )
            if registration_changed:
                await redirect_storage.set_client_info(
                    OAuthClientInformationFull(
                        client_id=client_id,
                        client_secret=client_secret,
                        client_name=client_metadata.client_name,
                        redirect_uris=client_metadata.redirect_uris,
                        scope=scope,
                        token_endpoint_auth_method=token_endpoint_auth_method,
                    )
                )
        yield OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=redirect_storage,
            redirect_handler=authorization.redirect_handler,
            callback_handler=authorization.callback_handler,
        )


@asynccontextmanager
async def build_http_client(
    server_config: object,
    *,
    auth_host: MCPAuthHost | None = None,
    auth_context: MCPAuthContext | None = None,
) -> AsyncIterator[httpx2.AsyncClient]:
    """Build and close an official-SDK-compatible HTTP client for one server.

    Pass a reusable ``auth_host`` when token reuse across Agent runs is needed.
    """

    server = _mapping(server_config)
    headers_value = server.get("headers")
    if headers_value is not None and not isinstance(headers_value, Mapping):
        raise ValueError("MCP HTTP headers must be a mapping")
    headers = {str(key): str(value) for key, value in cast(Mapping[object, object], headers_value or {}).items()}
    timeout_value = server.get("timeout")
    timeout = httpx2.Timeout(float(timeout_value)) if timeout_value is not None else None
    async with build_http_auth(server, auth_host=auth_host, auth_context=auth_context) as auth:
        client_kwargs: dict[str, Any] = {"follow_redirects": True}
        if headers:
            client_kwargs["headers"] = headers
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if auth is not None:
            client_kwargs["auth"] = auth
        async with httpx2.AsyncClient(**client_kwargs) as client:
            yield client


__all__ = [
    "BearerAuth",
    "EnvSecretResolver",
    "MCPAuthContext",
    "MCPAuthHost",
    "MCPAuthorizationCodeSession",
    "MCPTokenStorageNamespace",
    "NamespacedMemoryTokenStorage",
    "SecretResolver",
    "build_http_auth",
    "build_http_client",
    "build_token_storage_namespace",
    "canonicalize_mcp_url",
    "redact_headers",
    "redact_sensitive_data",
    "redact_url",
]
