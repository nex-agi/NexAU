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

"""对明文 HTTP 目标强制走 CONNECT 隧道。

## 为什么需要

HTTP 代理有两种用法，客户端按目标 scheme 自动二选一：

* `https://` 目标 → 先发 `CONNECT host:port`，凭据放在 CONNECT 请求头上，
  代理回 200 后隧道内自己做 TLS；
* `http://` 目标 → 不发 CONNECT，直接把 `GET http://host/path` 整条发给代理。

受控出网网关若把鉴权绑在 **CONNECT 会话**上（验完 token 后按客户端连接记账，
后续请求靠查会话表认人），第二种用法就没有可认的会话——无论请求头上带不带
凭据都会被拒。本模块让明文目标也先建隧道，隧道建成后在里面说明文 HTTP。

## 为什么用标准库而不是 httpx

httpx 的这个分支在 `httpcore.HTTPProxy.create_connection` 里按 scheme 硬分流，
且它的隧道连接类建完隧道后**无条件** `start_tls`，明文目标接不上。想复用就得
继承 `httpcore._sync.http_proxy` 这些私有模块，跨版本容易碎。而
`http.client.HTTPConnection.set_tunnel()` 恰好就是「CONNECT 之后说明文」的
标准库实现，零第三方耦合。

## 适用边界

只给**配置指定的固定端点**用（如 HTML 解析服务），不给用户输入的任意 URL 用：
受控出网部署里任意目标本来就该被策略拒，绕过 scheme 语义去够它不是本模块的事。
"""

from __future__ import annotations

import http.client
import json as _json
from typing import Any
from urllib.parse import urlsplit

import httpx

# 与 httpx 默认一致：不自动跟随重定向，由调用方决定
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def should_tunnel(target_url: str, proxy: httpx.Proxy | None) -> bool:
    """是否需要走本模块的强制隧道。

    三个条件缺一不可，任一不满足都应当退回普通 httpx 路径：

    * 配了代理——没配代理时是直连，不存在 CONNECT 这回事；
    * 目标是 `http://`——`https://` 目标 httpx 自己就会发 CONNECT；
    * 代理本身是 `http://`——`http.client.HTTPConnection` 到代理这一跳是明文的，
      代理若要求 TLS 得换 `HTTPSConnection`，当前没有这种部署，先明确不接管。
    """
    if proxy is None:
        return False
    if urlsplit(target_url).scheme != "http":
        return False
    return proxy.url.scheme == "http"


def post_json(
    target_url: str,
    proxy: httpx.Proxy,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, bytes]:
    """经 CONNECT 隧道向明文 HTTP 目标 POST 一个 JSON，返回 `(状态码, 响应体)`。

    `proxy.headers` 原样作为 CONNECT 请求头发出（受控网关要校验的
    `Proxy-Authorization` 就在里面），`headers` 则是隧道内那条业务请求的头。

    调用方负责兜异常：连接失败、代理拒绝、超时都会照常抛出。
    """
    target = urlsplit(target_url)
    if not target.hostname:
        raise ValueError(f"target url has no host: {target_url}")
    path = target.path or "/"
    if target.query:
        path = f"{path}?{target.query}"

    conn = http.client.HTTPConnection(
        proxy.url.host,
        proxy.url.port or 80,
        timeout=timeout,
    )
    try:
        # set_tunnel 的 headers 进的是 CONNECT 请求，不是隧道内那条业务请求
        conn.set_tunnel(
            target.hostname,
            target.port or 80,
            headers={k: v for k, v in proxy.headers.items()},
        )
        body = _json.dumps(payload).encode()
        # 不主动声明 Accept-Encoding：标准库不做透明解压，让服务端返回未压缩内容
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Host": target.netloc,
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                **headers,
            },
        )
        response = conn.getresponse()
        return response.status, response.read(_MAX_RESPONSE_BYTES)
    finally:
        conn.close()
