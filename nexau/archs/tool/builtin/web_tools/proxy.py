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

"""出网代理配置。

代理是**运行时级**能力，不属于任何一个工具：私有化部署里所有从沙箱发出的
HTTP 请求（聚合搜索、网页抓取、HTML 解析服务）都要经同一个受控出网网关，
凭同一份凭据过闸。所以配置读取放在这里，而不是跟着某个工具走。

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `NEXAU_HTTP_PROXY` | 否 | — | 显式正向代理地址，如 `http://gateway.internal:8444` |
| `NEXAU_HTTP_PROXY_AUTH` | 否 | — | 代理认证凭据，原样作为 `Proxy-Authorization` 发送 |

两个变量都没配时 `resolve_proxy()` 返回 `None`，httpx 按 `trust_env=True`
继续读标准 `HTTP_PROXY` / `HTTPS_PROXY`，既有部署行为不变。
"""

from __future__ import annotations

import os

import httpx

# 出网代理。刻意不用 SEARCH_ 前缀：代理是运行时级能力(见模块 docstring)，
# 且 SEARCH_* 是 aggregated_search.param_env_name() 的命名空间，占用会与参数级变量相撞。
PROXY_URL_ENV = "NEXAU_HTTP_PROXY"
PROXY_AUTH_ENV = "NEXAU_HTTP_PROXY_AUTH"


def resolve_proxy() -> httpx.Proxy | None:
    """解析显式代理配置，未配置返回 `None`。

    返回 `None` 时调用方把 `proxy=None` 交给 httpx，等价于不传——httpx 仍按
    `trust_env=True` 读标准 `HTTP_PROXY` / `HTTPS_PROXY`，**既有部署行为不变**。

    `PROXY_AUTH_ENV` 走 `httpx.Proxy(headers=...)`，httpx 会把它放到 **CONNECT
    隧道请求头**上。这正是受控出网网关要校验的位置：标准代理环境变量只能表达
    URL 里的 basic auth，带不了 `Bearer` token，所以必须单开一个变量。

    ⚠️ 只有 `https://` 目标才会走 CONNECT。`http://` 目标 httpx 用明文正向转发，
    认证头的落点与网关实现有关——受控网关若把鉴权绑在 CONNECT 会话上，明文目标
    会被拒。要经代理访问的内部服务应当提供 HTTPS 端点。
    """
    url = (os.getenv(PROXY_URL_ENV) or "").strip()
    if not url:
        return None

    auth = (os.getenv(PROXY_AUTH_ENV) or "").strip()
    if not auth:
        return httpx.Proxy(url)
    # 只支持 Proxy-Authorization 单个头：Rust 侧 reqwest 仅提供
    # `Proxy::custom_http_auth()`(固定写这一个头)，放开成任意 headers 会让
    # 两个实现无法对等，配置在一边能生效、另一边静默失效。
    return httpx.Proxy(url, headers={"Proxy-Authorization": auth})
