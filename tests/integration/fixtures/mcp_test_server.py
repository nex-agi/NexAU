"""Local official-SDK MCP server used by RFC-0029 black-box tests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlencode

from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.types import Receive, Scope, Send

mcp = MCPServer("nexau-rfc-0029-test")


@mcp.tool(description="Echo text and expose the explicitly configured test environment.")
def echo(text: str) -> dict[str, str]:
    """Return deterministic structured output for transport tests."""
    return {"text": text, "profile": os.environ.get("MCP_TEST_PROFILE", "missing")}


@mcp.tool(description="Add two integers.")
def add(left: int, right: int) -> int:
    """Return an integer result for concurrent call tests."""
    return left + right


class OAuthTestApplication:
    """OAuth test authorization server wrapped around the official MCP ASGI app."""

    def __init__(
        self,
        mcp_app: Callable[[Scope, Receive, Send], Awaitable[None]],
        *,
        issuer: str,
        mode: str,
        resource_path: str = "/mcp",
    ) -> None:
        self._mcp_app = mcp_app
        self._issuer = issuer
        self._mode = mode
        self._resource_path = resource_path
        self._challenge: str | None = None
        self.token_requests = 0
        self.refresh_requests = 0
        self.registration_requests = 0
        self.authorize_requests = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._mcp_app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path
        if path in {"/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"}:
            await JSONResponse(
                {
                    "resource": f"{self._issuer}{self._resource_path}",
                    "authorization_servers": [self._issuer],
                    "scopes_supported": ["tools:read", "tools:call"],
                }
            )(scope, receive, send)
            return
        if path == "/.well-known/oauth-authorization-server":
            await JSONResponse(
                {
                    "issuer": self._issuer,
                    "authorization_endpoint": f"{self._issuer}/authorize",
                    "token_endpoint": f"{self._issuer}/token",
                    "registration_endpoint": f"{self._issuer}/register",
                    "response_types_supported": ["code"],
                    "grant_types_supported": ["authorization_code", "refresh_token", "client_credentials"],
                    "token_endpoint_auth_methods_supported": ["none", "client_secret_basic"],
                    "code_challenge_methods_supported": ["S256"],
                    "authorization_response_iss_parameter_supported": True,
                }
            )(scope, receive, send)
            return
        if path == "/register":
            self.registration_requests += 1
            payload = await request.json()
            payload.update(
                {
                    "client_id": "dynamic-client",
                    "token_endpoint_auth_method": "none",
                }
            )
            await JSONResponse(payload, status_code=201)(scope, receive, send)
            return
        if path == "/authorize":
            self.authorize_requests += 1
            params = request.query_params
            if params.get("code_challenge_method") != "S256":
                await JSONResponse({"error": "invalid_request"}, status_code=400)(scope, receive, send)
                return
            self._challenge = params.get("code_challenge")
            redirect_uri = params["redirect_uri"]
            query = urlencode({"code": "test-code", "state": params["state"], "iss": self._issuer})
            await RedirectResponse(f"{redirect_uri}?{query}", status_code=302)(scope, receive, send)
            return
        if path == "/token":
            await self._handle_token(request, scope, receive, send)
            return
        if path == "/metrics":
            await JSONResponse(
                {
                    "token_requests": self.token_requests,
                    "refresh_requests": self.refresh_requests,
                    "registration_requests": self.registration_requests,
                    "authorize_requests": self.authorize_requests,
                }
            )(scope, receive, send)
            return

        if self._mode == "headers" and request.headers.get("x-test-tenant") != "north":
            await JSONResponse({"error": "missing_test_header"}, status_code=403)(scope, receive, send)
            return

        expected_token = {
            "bearer": "static-token",
            "client-credentials": "client-access",
            "authorization-code": "user-access-refresh" if self.refresh_requests else "user-access",
        }.get(self._mode)
        if expected_token is not None and request.headers.get("authorization") != f"Bearer {expected_token}":
            challenge = f'Bearer resource_metadata="{self._issuer}/.well-known/oauth-protected-resource", scope="tools:read tools:call"'
            await Response(status_code=401, headers={"WWW-Authenticate": challenge})(scope, receive, send)
            return
        await self._mcp_app(scope, receive, send)

    async def _handle_token(self, request: Request, scope: Scope, receive: Receive, send: Send) -> None:
        form = parse_qs((await request.body()).decode("utf-8"))
        grant_type = form.get("grant_type", [""])[0]
        self.token_requests += 1

        if grant_type == "client_credentials":
            credentials = request.headers.get("authorization", "").removeprefix("Basic ")
            try:
                decoded = base64.b64decode(credentials).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                decoded = ""
            if decoded != "service-client:service-secret":
                await JSONResponse({"error": "invalid_client"}, status_code=401)(scope, receive, send)
                return
            response: dict[str, Any] = {
                "access_token": "client-access",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        elif grant_type == "authorization_code":
            verifier = form.get("code_verifier", [""])[0]
            computed = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
            if form.get("code", [""])[0] != "test-code" or computed != self._challenge:
                await JSONResponse({"error": "invalid_grant"}, status_code=400)(scope, receive, send)
                return
            response = {
                "access_token": "user-access",
                "refresh_token": "refresh-token",
                "token_type": "Bearer",
                "expires_in": 0,
            }
        elif grant_type == "refresh_token":
            self.refresh_requests += 1
            if form.get("refresh_token", [""])[0] != "refresh-token":
                await JSONResponse({"error": "invalid_grant"}, status_code=400)(scope, receive, send)
                return
            response = {
                "access_token": "user-access-refresh",
                "refresh_token": "refresh-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        else:
            await JSONResponse({"error": "unsupported_grant_type"}, status_code=400)(scope, receive, send)
            return
        await JSONResponse(response)(scope, receive, send)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transport", choices=("stdio", "streamable-http", "sse"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stateless", action="store_true")
    parser.add_argument(
        "--auth",
        choices=("none", "headers", "bearer", "client-credentials", "authorization-code"),
        default="none",
    )
    args = parser.parse_args()

    pid_file = os.environ.get("MCP_TEST_PID_FILE")
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))

    if args.transport == "stdio":
        mcp.run("stdio")
    elif args.transport == "sse" and args.auth == "none":
        mcp.run("sse", host="127.0.0.1", port=args.port)
    elif args.auth == "none":
        mcp.run(
            "streamable-http",
            host="127.0.0.1",
            port=args.port,
            stateless_http=args.stateless,
        )
    else:
        import uvicorn

        issuer = f"http://127.0.0.1:{args.port}"
        if args.transport == "sse":
            mcp_app = mcp.sse_app(host="127.0.0.1")
            resource_path = "/sse"
        else:
            mcp_app = mcp.streamable_http_app(
                streamable_http_path="/mcp",
                stateless_http=args.stateless,
                host="127.0.0.1",
            )
            resource_path = "/mcp"
        app = OAuthTestApplication(mcp_app, issuer=issuer, mode=args.auth, resource_path=resource_path)
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
