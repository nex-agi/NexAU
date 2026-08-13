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

"""AggregatedWebSearch — 可切换服务商的聚合 Web 搜索工具。

高度参考 Nexau 内置 WebSearch(`nexau.archs.tool.builtin.web_tools` 的
`google_web_search` + `web_tool.SerperSearch`)，刻意保持一致的：

- gemini-cli 风格返回结构(`content` / `returnDisplay` / `sources` / `error`)
- 重试与指数退避策略(默认 3 次，`2 ** attempt` 秒退避)
- 错误归一方式(引擎层返回错误字符串，工具层包成 error dict，不抛异常)

两处实质差异：

1. **搜索服务商不再硬绑 Serper**，通过环境变量在 Serper / Seed / Baidu / XiaoBei 间切换；
2. **检索参数全部开放给调用方**(站点过滤、时间范围、权威度、行业、正文粒度…)，
   每个参数都有默认值，不传即退化成与内置 WebSearch 等价的行为。

# 术语：Provider vs Engine

这两层务必分清，环境变量也是按这两层切的：

- **Provider(服务商)** —— 调哪家的搜索 API：`Serper` / `Seed` / `Baidu` / `XiaoBei`。
- **Engine(底层搜索引擎)** —— 搜索结果实际来自哪个引擎：`google` / `bing` / `baidu`。
  只有聚合型 Provider(目前仅 `XiaoBei`)能选；其余 Provider 的引擎是固定的
  (Serper=Google、Seed=豆包自有索引、Baidu=百度)。

# 环境变量

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `SEARCH_PROVIDER` | 否 | `Serper` | 服务商：`Serper` / `Seed` / `Baidu` / `XiaoBei`，大小写不敏感 |
| `SEARCH_API_KEY` | 是 | — | 当前服务商的密钥 |
| `SEARCH_ENGINE` | 否 | 空(用服务商默认) | 底层搜索引擎，如 `google`、`google\\|baidu`；仅聚合型服务商生效 |
| `SEARCH_BASE_URL` | 否 | 各服务商内置 | 覆盖上游地址(私有化 / 内网部署) |
| `SEARCH_TIMEOUT` | 否 | `30` | 单次请求超时秒数 |
| `SEARCH_MAX_RETRIES` | 否 | `3` | **总**尝试次数(含首次)，下限 1 |

**上表就是全部搜索相关环境变量，每项只有唯一一个变量名**，密钥四家共用
`SEARCH_API_KEY`，没有按服务商区分的前缀变量。换服务商只需要改 `SEARCH_PROVIDER`
+ `SEARCH_API_KEY`。

# 出网代理(可选)

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `NEXAU_HTTP_PROXY` | 否 | — | 显式正向代理地址，如 `http://gateway.internal:8444` |
| `NEXAU_HTTP_PROXY_AUTH` | 否 | — | 代理认证凭据，原样作为 `Proxy-Authorization` 发送 |

**不配置时行为完全不变**——httpx 默认 `trust_env=True`，照旧读标准的
`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`。只有**代理需要自定义认证头**这一种场景
才用得上这两个变量：标准环境变量只能把凭据写进 URL(basic auth)，带不了 `Bearer`
这类 token，而受控出网网关普遍要求后者。

⚠️ `NEXAU_HTTP_PROXY_AUTH` 的值发送在 **CONNECT 隧道请求头**上(HTTPS 目标一律走
CONNECT)，不是业务请求头——网关校验的就是这一个。值要写完整、含认证方案前缀，
例如 `Bearer eyJhbGci...`。

⚠️ 这两个变量**刻意不带 `SEARCH_` 前缀**：代理是运行时级能力，`web_read` 等其它
出网工具将来复用同一套配置，不该绑死在搜索命名空间里；同时也避开了
`param_env_name()` 的 `SEARCH_*` 命名空间(有单测守着不许相撞)。

`SEARCH_ENGINE` 的取值宽松解析，`google`、`google|bing`、`google,bing`、
`["google","bing"]` 都认；它只是**默认值**，工具入参 `search_engine` 会覆盖它。

`SEARCH_PROVIDER` 的取值除标准名外，还接受
`小北`/`北坡`/`Orchestrator`→XiaoBei、`豆包`→Seed 等中文/俗称写法。

# 服务商对照

| Provider | 上游 | 鉴权 |
|------|------|------|
| `Serper` | `https://google.serper.dev/{search_type}` | `X-API-KEY` 头 |
| `Seed` | 豆包搜索 Custom 版 `https://open.feedcoopapi.com/search_api/web_search` | `Authorization: Bearer` |
| `Baidu` | 百度 AI 搜索 `https://qianfan.baidubce.com/v2/ai_search/web_search` | `Authorization: Bearer` |
| `XiaoBei` | 北坡聚合搜索 `https://search.xiaobei.top/tools/web_research`(内网 `http://search.iqjzf.com`) | `X-API-Key` 头 |

# 官方文档

改参数、查值域、排错前先看对应上游的官方文档——本文件的注释是二手信息，
**上游文档才是事实源**：
- **Serper** —— 官网与文档 <https://serper.dev>；
  交互式调试台 <https://serper.dev/playground>；Key 管理 <https://serper.dev/api-keys>
- **Seed（豆包搜索）** —— Custom 版 <https://docs.volcengine.com/docs/87772/2272953>；
  Global 版 <https://docs.volcengine.com/docs/87772/2548026>；控制台 <https://console.volcengine.com/>
- **Baidu（百度 AI 搜索）** —— 「百度搜索」API <https://cloud.baidu.com/doc/qianfan-api/s/Wmbq4z7e5>；
  API Key 说明 <https://cloud.baidu.com/doc/BAIDU_AI_SEARCH/s/5mkmgi38d>；
  控制台 <https://console.bce.baidu.com/ai-search/home>
- **XiaoBei（北坡聚合搜索）** —— 文档 <https://search.xiaobei.top/docs>；
  OpenAPI <https://search.xiaobei.top/openapi.json>

Google 检索算子(`site:` / `-site:` / `tbs` 时效参数)的说明见
<https://support.google.com/websearch/answer/2466433>。

⚠️ 豆包的文档页是 SPA，正文以 Quill delta JSON 嵌在 HTML 里，
用普通抓取工具只能拿到导航——**请用浏览器打开**。

# 参数支持度矩阵

各引擎能力不同，**不支持的参数会被静默忽略**(不会报错，也不会改变语义)：

| 参数 | Serper | Seed | Baidu | XiaoBei |
|------|:------:|:----:|:-----:|:-------:|
| `num_results` | ✅ `num` | ✅ `Count` | ✅ `top_k` | ⚠️ 每引擎量级，本地再截断 |
| `search_type` | ✅ 六端点 | ⚠️ 仅 web/image | ⚠️ 仅影响时效 | ⚠️ 仅网页 |
| `time_range` 预设 | ✅ `tbs=qdr:*` | ✅ 原生 | ✅ `search_recency_filter` | ✅ `day/week/month/year` |
| `time_range` 自定义区间 | ✅ `tbs=cdr:*` | ✅ 原生 | ✅ `search_filter.range.page_time` | ❌ |
| `sites` | ✅ 算子 | ✅ `Filter.Sites` | ✅ `match.site` | ✅ 算子+本地 |
| `block_hosts` | ✅ 算子 | ✅ `Filter.BlockHosts` | ⚠️ 上游无效，走本地过滤 | ✅ 算子+本地 |
| `authority_only` | ❌ | ✅ `AuthInfoLevel` | ❌ | ❌ |
| `industry` | ❌ | ✅ `Industry` | ❌ | ❌ |
| `query_rewrite` | ✅ `autocorrect` | ✅ `QueryRewrite` | ❌ | ❌ |
| `need_content` | ❌ | ✅ `NeedContent` | ❌ | ❌ |
| `full_content` | ❌ | ✅ `Content` | ⚠️ no-op(实测 `snippet` 与 `content` 内容相同) | ✅ `scrape_top_n` 抓正文 |
| `content_format` | ❌ | ✅ `ContentFormats` | ❌ | ❌ (固定 markdown) |
| `country` / `location` / `page` | ✅ `gl` / `location` / `page` | ❌ | ❌ | ❌ |
| `language` | ✅ `hl` | ❌ | ❌ | ✅ 原生(BCP-47) |
| `search_engine` | ❌ | ❌ | ❌ | ✅ `engines` 选下游子集 |
| `render_js` | ❌ | ❌ | ❌ | ✅ `fast_mode` 取反 |
| `max_content_chars` | ✅ 本地截断 | ✅ 本地截断 | ✅ 本地截断 | ✅ 上游+本地双重 |

# 参数覆盖度（哪些上游能力**有意没有**暴露）

不是全量透传。对账口径：XiaoBei 按其 OpenAPI schema，Seed 按豆包 Custom 版官方
文档全文，Serper 按实测请求回显。下面这些是**有意省略**的，不是遗漏：

| 上游 | 未暴露字段 | 原因 |
|------|-----------|------|
| Seed | `NeedSummary` | 绑定 `web_summary` 搜索类型，官方 2026-06-23 起不再支持新增开通 |
| Seed | `ImageWidth/Height{Max,Min}`、`ImageShapes` | 只在 `search_type=images` 下有意义，属长尾；需要时再加 |
| Baidu | `safe_search`、`config_id`、`search_filter.geo/image` | 长尾；`geo.city` 语义与本工具 `location` 不同，硬映射会错 |

另有几个字段是**刻意收归内部**、不该由调用方决定的：
`Seed.Filter.NeedUrl`(恒 true，保证结果有可点链接)、
`XiaoBei.wait_for_result` / `timeout_s`(同步/异步编排与超时预算)、
`XiaoBei.scrape_top_n`(由 `full_content` 派生)。

Serper 侧 9 个请求字段已 100% 覆盖。

# 未知参数的处理

**服务商不支持的参数一律静默忽略，工具调用照常成功**——不会因此报错。
其中"能选而选错"的情形会打 `warning` 日志留痕(如给 Serper 传 `search_engine`、
给 XiaoBei 传不存在的引擎名)，避免出现"我明明配了却没生效"这种无从排查的困惑。

过度收窄条件导致 0 条结果时，返回的是**成功且 `sources: []`**
(content 为 "No results found.")，不是 error——让模型知道"搜过了但没有"，
而不是把它误当成故障去重试。

# 返回结构

成功::

    {
      "content": 'Web search results for "LLM 新闻" (provider: XiaoBei):\\n\\n[1] 标题 (https://…)\\n    摘要',
      "returnDisplay": 'Search results for "LLM 新闻" via XiaoBei returned (5 results).',
      "sources": [{"title": …, "link": …, "snippet": …,
                   "provider": "XiaoBei", "engine": "baidu"}, …],
      "provider": "XiaoBei"
    }

失败::

    {"content": "Error: …", "returnDisplay": "Error performing web search.", "error": {"message": "…", "type": "WEB_SEARCH_FAILED"}}
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from .proxy import PROXY_AUTH_ENV, PROXY_URL_ENV, resolve_proxy

logger = logging.getLogger(__name__)

__all__ = ["PROXY_AUTH_ENV", "PROXY_URL_ENV", "resolve_proxy", "web_search"]


def _host_of(url: str) -> str:
    """取 URL 的 host(小写、去端口)，取不到返回空串。"""
    try:
        return (urlsplit(url).netloc or "").lower().split(":")[0]
    except ValueError:
        return ""


def _host_matches(host: str, pattern: str) -> bool:
    """host 是否命中域名模式：精确相等或是其子域。"""
    return bool(host) and (host == pattern or host.endswith("." + pattern))


def _apply_site_operators(query: str, sites: list[str], block_hosts: list[str]) -> str:
    """把站点白/黑名单编译成 Google 系查询词算子。

    给没有结构化站点过滤字段的上游用(Serper、北坡聚合搜索)。
    实测收益很大：北坡上纯靠本地过滤 `arxiv.org` 只能捞到 1/28 条，
    把 `site:arxiv.org` 拼进查询词后是 20/20。
    """
    parts = [query]
    if sites:
        parts.append("(" + " OR ".join(f"site:{s}" for s in sites) + ")")
    parts.extend(f"-site:{h}" for h in block_hosts)
    return " ".join(parts)


# ---------------------------------------------------------------- 默认值常量
# 这些默认值必须与 tools/AggregatedWebSearch.tool.yaml 里的 `default` 一一对应，
# 改一处就要改另一处，否则「不传参」和「传默认值」的行为会分叉。

DEFAULT_NUM_RESULTS = 10
DEFAULT_SEARCH_TYPE = "search"
DEFAULT_TIME_RANGE = ""  # 空串 = 不做时效限制
DEFAULT_SITES = ""  # 空串 = 不限定站点
DEFAULT_BLOCK_HOSTS = ""  # 空串 = 不屏蔽站点
DEFAULT_AUTHORITY_ONLY = False
DEFAULT_INDUSTRY = ""  # 空串 = 不做行业限定
DEFAULT_QUERY_REWRITE = False
DEFAULT_NEED_CONTENT = False
DEFAULT_FULL_CONTENT = False
DEFAULT_CONTENT_FORMAT = "text"
DEFAULT_MAX_CONTENT_CHARS = 1000
DEFAULT_COUNTRY = ""  # 空串 = 跟随上游默认区域
DEFAULT_LANGUAGE = ""  # 空串 = 跟随上游默认语言
DEFAULT_LOCATION = ""  # 空串 = 不做地理定位
DEFAULT_PAGE = 1
DEFAULT_SEARCH_ENGINE = ""  # 空串 = 跟随 SEARCH_ENGINE 环境变量，再缺省就用上游默认
DEFAULT_RENDER_JS = False  # 抓正文默认走 HTTP 快速引擎，不开浏览器渲染

# 服务商级默认值(走环境变量覆盖)
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3

# ---------------------------------------------------------------- 参数级环境变量
# **每个工具参数都能用环境变量设部署级默认值**，命名规则：
#
#     SEARCH_ + 参数名大写；参数名自带 `search_` 前缀的去重
#     content_format → SEARCH_CONTENT_FORMAT
#     num_results    → SEARCH_NUM_RESULTS
#     search_engine  → SEARCH_ENGINE      (不是 SEARCH_SEARCH_ENGINE)
#     search_type    → SEARCH_TYPE        (同上)
#
# 生效优先级：**调用方显式传参 > 环境变量 > 下表的内置默认值**。
#
# 之所以要能区分"没传"与"显式传了默认值"：实测模型会把一堆默认值原样回传
# (如 content_format="text")，若按"值等于默认就用环境变量"来判，部署方设的
# 默认会时灵时不灵。因此工具函数签名一律用 None 哨兵，None 才表示"没传"。
PARAM_DEFAULTS: dict[str, Any] = {
    "num_results": DEFAULT_NUM_RESULTS,
    "search_type": DEFAULT_SEARCH_TYPE,
    "time_range": DEFAULT_TIME_RANGE,
    "sites": DEFAULT_SITES,
    "block_hosts": DEFAULT_BLOCK_HOSTS,
    "authority_only": DEFAULT_AUTHORITY_ONLY,
    "industry": DEFAULT_INDUSTRY,
    "query_rewrite": DEFAULT_QUERY_REWRITE,
    "need_content": DEFAULT_NEED_CONTENT,
    "full_content": DEFAULT_FULL_CONTENT,
    "content_format": DEFAULT_CONTENT_FORMAT,
    "max_content_chars": DEFAULT_MAX_CONTENT_CHARS,
    "country": DEFAULT_COUNTRY,
    "language": DEFAULT_LANGUAGE,
    "location": DEFAULT_LOCATION,
    "page": DEFAULT_PAGE,
    "search_engine": DEFAULT_SEARCH_ENGINE,
    "render_js": DEFAULT_RENDER_JS,
}

# 服务商级环境变量，参数级命名规则不得与之相撞(有单测守着)
PROVIDER_ENV_KEYS = frozenset(
    {
        "SEARCH_PROVIDER",
        "SEARCH_API_KEY",
        "SEARCH_BASE_URL",
        "SEARCH_TIMEOUT",
        "SEARCH_MAX_RETRIES",
    }
)

# 布尔环境变量的真值写法(其余一律按 False)
_TRUE_LITERALS = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSE_LITERALS = frozenset({"0", "false", "no", "off", "n", "f"})

# 出网代理定义在 `.proxy`：它是运行时级能力，网页抓取 / HTML 解析服务同样要用，
# 不该由聚合搜索独占。这里 re-export 保持 `aggregated_search.resolve_proxy` 可用。


def param_env_name(param: str) -> str:
    """工具参数名 -> 对应的环境变量名。

    `SEARCH_` + 参数名大写；参数名已带 `search_` 前缀的去重，
    避免出现 `SEARCH_SEARCH_ENGINE` 这种叠词。
    """
    stem = param[len("search_") :] if param.startswith("search_") else param
    return f"SEARCH_{stem.upper()}"


def _coerce_env_value(raw: str, default: Any, env_name: str) -> Any:
    """按内置默认值的类型解析环境变量字符串。

    解析不了就退回内置默认并留一条 warning——
    一个写错的可选调优变量不该把整次检索打挂。
    """
    text = raw.strip()
    if isinstance(default, bool):
        low = text.lower()
        if low in _TRUE_LITERALS:
            return True
        if low in _FALSE_LITERALS:
            return False
        logger.warning(
            "环境变量 %s=%r 不是合法布尔值(可用 true/false)，已回退默认 %r",
            env_name,
            raw,
            default,
        )
        return default
    if isinstance(default, int):
        try:
            return int(text)
        except ValueError:
            logger.warning("环境变量 %s=%r 不是整数，已回退默认 %r", env_name, raw, default)
            return default
    return text


def resolve_param(param: str, value: Any) -> Any:
    """按「入参 > 环境变量 > 内置默认」解析单个参数的最终取值。"""
    if value is not None:
        return value
    default = PARAM_DEFAULTS[param]
    env_name = param_env_name(param)
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return default
    return _coerce_env_value(raw, default, env_name)


# search_type=news 且调用方没显式指定 time_range 时，默认收敛到最近一周
NEWS_FALLBACK_TIME_RANGE = "OneWeek"

# 合法的时效枚举(与豆包搜索文档一致)；另支持 `YYYY-MM-DD..YYYY-MM-DD` 区间
TIME_RANGE_PRESETS = ("OneDay", "OneWeek", "OneMonth", "OneYear")


class SearchProviderError(Exception):
    """**不可重试**的错误：缺 key、引擎名非法、账号/套餐/额度/参数问题。

    这类问题重试多少次都一样，需要人去改配置或开通服务，
    因此基类会跳过退避直接上抛，工具层标成 `WEB_SEARCH_CONFIG_ERROR`。
    """


class RetryableUpstreamError(Exception):
    """**可重试**的上游临时故障：服务端内部错误、QPS 限流等。

    由基类按指数退避重试；重试耗尽后才降级成错误字符串。
    """


@dataclass
class SearchOptions:
    """一次搜索的全部可调参数。

    字段默认值与 `DEFAULT_*` 常量、tool YAML 的 `default` 三者保持同步。
    引擎不支持的字段会被该引擎静默忽略，详见模块 docstring 的支持度矩阵。

    每个字段的注释都标注了它在上游映射到哪个原生参数，
    **要查值域 / 上限 / 语义细节，请对照该上游的官方文档**(链接见模块 docstring
    的「官方文档」一节)。
    """

    # 返回结果条数。上限按服务商/类型钳制(豆包 web≤50、image≤5；北坡≤30)
    # → Serper `num` / 豆包 `Count` / 千帆 `top_k` / 北坡 `max_results`
    num_results: int = DEFAULT_NUM_RESULTS
    # 搜索类型：search 网页 / news 资讯 / images 图片 / places 地点 / videos / scholar
    # → Serper 是六个独立端点(见 https://serper.dev/playground)；豆包只有 web / image 两种 SearchType
    search_type: str = DEFAULT_SEARCH_TYPE
    # 时效过滤：OneDay / OneWeek / OneMonth / OneYear / YYYY-MM-DD..YYYY-MM-DD
    # → Serper 编译成 Google `tbs`(qdr:* 与 cdr:*，见
    #   https://support.google.com/websearch/answer/2466433)
    # → 豆包 `TimeRange`(枚举同名，原生支持日期区间)
    # → 千帆 `search_recency_filter` / 北坡 `time_range`(day/week/month/year)
    time_range: str = DEFAULT_TIME_RANGE
    # 限定站点，'|' 分隔完整域名，最多 20 个，如 "arxiv.org|nature.com"
    # → 豆包 `Filter.Sites`(原生)；Serper / 北坡降级成 `site:` 查询词算子
    sites: str = DEFAULT_SITES
    # 屏蔽站点，'|' 分隔完整域名，最多 5 个
    # → 豆包 `Filter.BlockHosts`(原生)；Serper / 北坡降级成 `-site:` 算子
    block_hosts: str = DEFAULT_BLOCK_HOSTS
    # 只要"非常权威"来源(政府/官媒/权威机构)，会显著减少结果数
    # → 豆包 `Filter.AuthInfoLevel=1`。权威度分级定义见豆包文档「站点权威度分级说明」
    authority_only: bool = DEFAULT_AUTHORITY_ONLY
    # 行业垂直检索：finance 金融 / game 电子游戏 / gov 政务
    # → 豆包 `Filter.Industry`(枚举以豆包 Custom 版文档为准)
    industry: str = DEFAULT_INDUSTRY
    # 是否让上游改写查询词(提召回，但增加耗时)
    # → Serper `autocorrect`(Google 拼写纠正) / 豆包 `QueryControl.QueryRewrite`
    query_rewrite: bool = DEFAULT_QUERY_REWRITE
    # 是否只保留"抓到了正文"的结果(过滤空壳页)
    # → 豆包 `Filter.NeedContent`
    need_content: bool = DEFAULT_NEED_CONTENT
    # 用完整正文替代摘要(信息更全，但极耗上下文)
    # → 豆包取 `WebItem.Content` 而非 `Summary`；北坡触发 `scrape_top_n` 抓取
    full_content: bool = DEFAULT_FULL_CONTENT
    # 正文格式：text / markdown，仅 full_content=True 时有意义
    # → 豆包 `ContentFormats`
    content_format: str = DEFAULT_CONTENT_FORMAT
    # 单条摘要/正文的截断长度，防止把上下文撑爆(本工具本地截断，非上游参数)
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS
    # 检索区域(ISO 3166-1 两位国家码，如 cn / us)
    # → Serper `gl`，取值可在 https://serper.dev/playground 里选
    country: str = DEFAULT_COUNTRY
    # 结果语言(BCP-47，如 zh-cn / en)
    # → Serper `hl` / 北坡 `language`
    language: str = DEFAULT_LANGUAGE
    # 地理定位描述串(如 "Tokyo, Japan")，本地化检索/地点搜索用
    # → Serper `location`；合法取值可在 https://serper.dev/playground 里试
    location: str = DEFAULT_LOCATION
    # 结果页码，从 1 开始；翻页取更靠后的结果
    # → Serper `page`
    page: int = DEFAULT_PAGE
    # 底层搜索引擎，'|' 分隔，如 "google|baidu"；空则用 SEARCH_ENGINE 环境变量
    # → 北坡 `options.engines`(见 https://search.xiaobei.top/docs)
    search_engine: str = DEFAULT_SEARCH_ENGINE
    # 抓正文时是否用浏览器渲染(能拿到 JS 动态生成的内容，但明显更慢)
    # → 北坡 `fast_mode`(取反：render_js=True 即 fast_mode=False)
    render_js: bool = DEFAULT_RENDER_JS

    def __post_init__(self) -> None:
        # search_type=news 是"语义意图"而非豆包原生类型，
        # 调用方没显式给 time_range 时替它收敛到最近一周
        if self.search_type == "news" and not self.time_range:
            self.time_range = NEWS_FALLBACK_TIME_RANGE
        # 防御性下限，避免 0 / 负数传到上游
        self.num_results = max(1, int(self.num_results))
        self.max_content_chars = max(100, int(self.max_content_chars))
        self.page = max(1, int(self.page))

    def parse_custom_range(self) -> tuple[str, str] | None:
        """把 `YYYY-MM-DD..YYYY-MM-DD` 拆成 (起, 止)；不是该形态则返回 None。"""
        if ".." not in self.time_range:
            return None
        start, _, end = self.time_range.partition("..")
        start, end = start.strip(), end.strip()
        return (start, end) if start and end else None

    def as_dict(self) -> dict[str, Any]:
        """按字段名取值的快照。

        用于「本服务商忽略了哪些参数」的比对——仓库规范禁止 `getattr` 动态属性访问
        （见 CLAUDE.md「Type Safety Guidelines」），因此走 dataclass 的静态字段。
        """
        return asdict(self)

    def split_sites(self) -> list[str]:
        """限定站点列表(已去空)。"""
        return [s.strip() for s in self.sites.split("|") if s.strip()]

    def split_block_hosts(self) -> list[str]:
        """屏蔽站点列表(已去空)。"""
        return [s.strip() for s in self.block_hosts.split("|") if s.strip()]

    def split_engines(self) -> list[str]:
        """底层搜索引擎列表(已去空、小写)。

        宽松解析，`google`、`google|bing`、`google,bing`、`["google","bing"]`
        都能认——环境变量与 LLM 生成的入参格式都不完全可控。
        """
        raw = self.search_engine.strip().strip("[]")
        return [part.strip().strip("\"'").lower() for part in re.split(r"[|,\s]+", raw) if part.strip().strip("\"'")]


class SearchProviderBase(ABC):
    """搜索引擎适配器基类。

    子类只需实现 `_do_search`(单次请求 + 结果归一)，
    重试、退避、异常归一、截断都在基类 `search` 里统一处理。
    """

    # 引擎显示名，会写进每条结果的 `engine` 字段
    name: str = "base"
    # 未配置 SEARCH_BASE_URL 时使用的上游地址
    default_base_url: str = ""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise SearchProviderError(
                f"{self.name} API key is required. 请设置环境变量 SEARCH_API_KEY(或沿用旧变量 {_LEGACY_API_KEY_ENV})。"
            )
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        # 语义是**总尝试次数**(与既有 web_tool.SerperSearch 一致，错误文案也写
        # "after N attempts")。必须钳到 >=1：`range(0)` 会让循环体一次都不执行，
        # `_do_search` 永不被调用，工具在不发出任何请求的情况下 100% 失败。
        self.max_retries = max(1, max_retries)

    # ---------------------------------------------------------------- 子类实现

    @abstractmethod
    def _do_search(
        self,
        client: httpx.Client,
        query: str,
        options: SearchOptions,
    ) -> list[dict[str, Any]]:
        """发起一次搜索请求，返回**已归一**的结果列表。

        归一后每项至少含 `title` / `link` / `snippet`，
        允许附带 `date` / `source` / `position` / `authority` 等可选字段。
        """

    # ---------------------------------------------------------------- 公共流程

    # 该服务商是否支持"选择底层搜索引擎"(只有聚合型服务商支持)
    supports_engine_choice: bool = False
    # 该服务商是否支持"抓正文时用浏览器渲染"
    supports_render_js: bool = False
    # 本服务商**不支持、会被忽略**的参数名。设了非默认值就告警——
    # "配了却没生效"必须留痕，否则是最难排查的一类问题
    IGNORED_PARAMS: frozenset[str] = frozenset()

    def search(self, query: str, options: SearchOptions) -> list[dict[str, Any]] | str:
        """带重试的搜索入口。

        成功返回结果列表；失败返回错误字符串(而不是抛异常)——
        这是内置 `SerperSearch.search` 的既有约定，工具层依赖它区分成功/失败。
        """
        # 指定了底层引擎但当前服务商不支持选择时，明确留痕——
        # 静默忽略会让"我明明配了 SEARCH_ENGINE=baidu"变成无从排查的疑惑
        if options.split_engines() and not self.supports_engine_choice:
            logger.warning(
                "%s 不支持选择底层搜索引擎，search_engine=%r 已忽略",
                self.name,
                options.search_engine,
            )
        supplied = options.as_dict()
        for param in sorted(self.IGNORED_PARAMS):
            value = supplied.get(param)
            if value != PARAM_DEFAULTS.get(param):
                logger.warning("%s 不支持 %s，本次传入的 %r 已忽略", self.name, param, value)
        if options.render_js and not self.supports_render_js:
            logger.warning("%s 不支持浏览器渲染，render_js=True 已忽略", self.name)
        elif options.render_js and not options.full_content:
            # 渲染只发生在抓正文阶段；没开 full_content 就压根不会抓页面，
            # 这时 render_js 是死设置——不留痕就会变成"我明明开了却没用"
            logger.warning(
                "%s: render_js=True 但 full_content=False，本次不抓正文，渲染设置无效",
                self.name,
            )

        timeout = httpx.Timeout(
            connect=self.timeout,
            read=self.timeout,
            write=self.timeout,
            pool=self.timeout,
        )

        # 每次 search 重新解析：允许运行期改配置(轮换凭据)后下一次调用即生效，
        # 不需要重建 provider 实例。未配置时为 None，httpx 行为与改动前一致。
        proxy = resolve_proxy()

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=timeout, proxy=proxy) as client:
                    results = self._do_search(client, query, options)
                # 统一补引擎标识并按 max_content_chars 截断
                for item in results:
                    item["provider"] = self.name
                    snippet = item.get("snippet")
                    if isinstance(snippet, str) and len(snippet) > options.max_content_chars:
                        item["snippet"] = snippet[: options.max_content_chars] + "..."
                return results[: options.num_results]

            except SearchProviderError:
                # 配置 / 账号类错误重试没有意义，直接上抛给工具层
                raise

            except RetryableUpstreamError as e:
                # 上游临时故障，退避后重试
                if attempt == self.max_retries - 1:
                    return f"Upstream error after {self.max_retries} attempts: {str(e)}"
                time.sleep(2**attempt)

            except httpx.ConnectTimeout as e:
                if attempt == self.max_retries - 1:
                    return f"Connection timeout after {self.max_retries} attempts: {str(e)}"
                time.sleep(2**attempt)  # 指数退避

            except httpx.TimeoutException as e:
                if attempt == self.max_retries - 1:
                    return f"Request timeout after {self.max_retries} attempts: {str(e)}"
                time.sleep(2**attempt)

            except httpx.HTTPStatusError as e:
                # 4xx 里除 429 外都是确定性错误，重试只会浪费时间
                status = e.response.status_code
                detail = e.response.text[:300]
                if 400 <= status < 500 and status != 429:
                    return f"HTTP error {status}: {detail}"
                if attempt == self.max_retries - 1:
                    return f"HTTP error {status}: {detail}"
                time.sleep(2**attempt)

            except Exception as e:  # noqa: BLE001 — 归一为错误字符串，与内置行为一致
                if attempt == self.max_retries - 1:
                    return f"Unexpected error: {type(e).__name__}: {str(e)}"
                time.sleep(2**attempt)

        return f"Failed to complete search after {self.max_retries} attempts"


class SerperProvider(SearchProviderBase):
    """Serper(google.serper.dev)——与 Nexau 内置 WebSearch 同源的实现。

    Serper 没有站点/权威度这类结构化过滤字段，因此 `sites` / `block_hosts`
    降级成 Google 查询词算子(`site:` / `-site:`)，`time_range` 映射成 `tbs`。

    官方文档:
        - 官网与文档: https://serper.dev
        - 交互式调试(能直接看各参数对结果的影响): https://serper.dev/playground
        - API Key 管理: https://serper.dev/api-keys
        - Google `tbs` 时效算子: https://support.google.com/websearch/answer/2466433
    """

    name = "Serper"
    default_base_url = "https://google.serper.dev"
    IGNORED_PARAMS = frozenset({"authority_only", "industry", "need_content", "full_content", "content_format"})

    # search_type -> (URL 路径, 响应里承载结果的字段名)
    # 多数端点用同名字段，但 scholar 走 `organic`，所以路径和字段要分开记
    ENDPOINT_FOR_TYPE = {
        "search": ("search", "organic"),
        "news": ("news", "news"),
        "images": ("images", "images"),
        "places": ("places", "places"),
        "videos": ("videos", "videos"),
        "scholar": ("scholar", "organic"),
    }
    # time_range 预设 -> Google `tbs` 时效算子
    TBS_FOR_TIME_RANGE = {
        "OneDay": "qdr:d",
        "OneWeek": "qdr:w",
        "OneMonth": "qdr:m",
        "OneYear": "qdr:y",
    }

    @staticmethod
    def _to_google_date(iso_date: str) -> str:
        """`YYYY-MM-DD` -> Google `tbs` 要的 `M/D/YYYY`(不补前导零)。"""
        y, m, d = iso_date.split("-")
        return f"{int(m)}/{int(d)}/{int(y)}"

    def _build_tbs(self, options: SearchOptions) -> str | None:
        """把 time_range 编译成 Google `tbs` 算子。

        预设枚举走 `qdr:*`；`YYYY-MM-DD..YYYY-MM-DD` 走自定义区间
        `cdr:1,cd_min:…,cd_max:…`(已实测生效)。
        """
        preset = self.TBS_FOR_TIME_RANGE.get(options.time_range)
        if preset:
            return preset
        custom = options.parse_custom_range()
        if not custom:
            return None
        try:
            start, end = (self._to_google_date(d) for d in custom)
        except ValueError:
            # 日期格式不合法就当没传，不因为一个可选参数把整次检索打挂
            logger.warning("无法解析 time_range=%r，已忽略", options.time_range)
            return None
        return f"cdr:1,cd_min:{start},cd_max:{end}"

    def _do_search(
        self,
        client: httpx.Client,
        query: str,
        options: SearchOptions,
    ) -> list[dict[str, Any]]:
        # 1. 取端点与结果字段
        endpoint_conf = self.ENDPOINT_FOR_TYPE.get(options.search_type)
        if endpoint_conf is None:
            raise SearchProviderError(
                f"Invalid search type: {options.search_type}. Serper search type should be one of {list(self.ENDPOINT_FOR_TYPE)}"
            )
        path, result_key = endpoint_conf

        # 2. Serper 没有结构化站点过滤，降级成 Google 查询词算子
        payload: dict[str, Any] = {
            "q": _apply_site_operators(query, options.split_sites(), options.split_block_hosts()),
            "num": options.num_results,
            # query_rewrite 在 Serper 侧就是 Google 的拼写纠正
            "autocorrect": options.query_rewrite,
        }
        # 3. 逐个映射可选参数，空值一律不发，避免覆盖上游默认
        tbs = self._build_tbs(options)
        if tbs:
            payload["tbs"] = tbs
        if options.country:
            payload["gl"] = options.country
        if options.language:
            payload["hl"] = options.language
        if options.location:
            payload["location"] = options.location
        if options.page > 1:
            payload["page"] = options.page

        # 4. 发起请求
        response = client.post(
            f"{self.base_url}/{path}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        parsed: dict[str, Any] = response.json()
        raw: list[dict[str, Any]] = parsed.get(result_key) or []

        # 5. places 结构与网页结果完全不同(没有 link/snippet，只有坐标与评分)，单独归一
        if options.search_type == "places":
            return [self._normalize_place(item) for item in raw[: options.num_results]]

        # 6. 通用归一，并丢掉 base64 内联图(会把上下文撑爆)
        results: list[dict[str, Any]] = []
        for item in raw[: options.num_results]:
            if str(item.get("imageUrl", "")).startswith("data:"):
                item.pop("imageUrl", None)
            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "link": item.get("link") or item.get("url", ""),
                    "snippet": item.get("snippet") or item.get("description", ""),
                    "date": _clean_date(item.get("date")),
                    "source": item.get("source"),
                    "position": item.get("position"),
                }
            )
        return results

    @staticmethod
    def _normalize_place(item: dict[str, Any]) -> dict[str, Any]:
        """地点结果归一。

        Serper 的 places 条目**没有 link 字段**，只有 `cid`(Google Maps 商户 ID)。
        直接套通用归一会产出空链接，因此这里用 cid 拼出可点击的地图链接，
        并把地址 / 分类 / 评分揉进 snippet——否则模型拿到的是一堆没有上下文的店名。
        """
        cid = item.get("cid")
        link = item.get("website") or (f"https://www.google.com/maps?cid={cid}" if cid else "")

        bits = [item.get("address"), item.get("category")]
        rating = item.get("rating")
        if rating:
            count = item.get("ratingCount")
            bits.append(f"评分 {rating}" + (f"({count} 条)" if count else ""))
        if item.get("phoneNumber"):
            bits.append(str(item["phoneNumber"]))

        return {
            "title": item.get("title", "Untitled"),
            "link": link,
            "snippet": " · ".join(b for b in bits if b),
            "date": None,
            "source": item.get("category"),
            "position": item.get("position"),
        }


class SeedProvider(SearchProviderBase):
    """Seed(豆包搜索 Custom 版，原「联网搜索 / 融合信息搜索」)。

    上游: `POST https://open.feedcoopapi.com/search_api/web_search`
    鉴权: `Authorization: Bearer <API_KEY>`

    官方文档(⚠️ SPA 页面，正文是 Quill delta JSON，请用浏览器打开):
        - Custom 版 API 参考(本类实现依据，含全部请求/响应字段与错误码):
          https://docs.volcengine.com/docs/87772/2272953
        - Global 版 API 参考: https://docs.volcengine.com/docs/87772/2548026
        - 控制台(建 API Key / 查额度与套餐): https://console.volcengine.com/

    计费与限流(摘自文档，以文档为准): 账号维度默认 5 QPS 可提工单扩容；
    每个火山账号每月 500 次免费额度，与 Global 版共用、优先消耗。

    两个必须照做的约定(照抄自官方文档，踩过才知道)：

    1. **失败也返回 HTTP 200**，错误藏在 `ResponseMetadata.Error`
       (`CodeN` / `Code` / `Message`)。只看 HTTP 状态码会把失败当成功。
    2. **`Snippet` 不能喂大模型**。文档明确写着 `Snippet` 约 200 字，
       "字数限制导致缺失相关信息，仅建议用于搜索结果的列表展示，强烈不建议
       用于大模型场景"；应当使用 `Summary`(500~1000 字，"推荐用于大模型场景")。
       因此归一时优先取 `Summary`；调用方显式要 `full_content` 时才取 `Content`。
    """

    name = "Seed"
    default_base_url = "https://open.feedcoopapi.com"
    IGNORED_PARAMS = frozenset({"country", "language", "location", "page"})

    # Query 长度上限(文档：1~100 字符，过长会被上游截断)
    MAX_QUERY_CHARS = 100
    # Count 上限(文档：web 最多 50 条；image 最多 5 条)
    MAX_COUNT_WEB = 50
    MAX_COUNT_IMAGE = 5
    # 站点过滤数量上限(文档：Sites 最多 20 个，BlockHosts 最多 5 个)
    MAX_SITES = 20
    MAX_BLOCK_HOSTS = 5

    # search_type -> 豆包 SearchType。
    # 豆包只有 web / image 两种原生类型，其余语义靠退化 + TimeRange 近似。
    SEARCH_TYPE_MAP = {
        "search": "web",
        "news": "web",  # 无独立新闻频道，靠 TimeRange 收敛时效
        "places": "web",
        "images": "image",
        "videos": "web",
        "scholar": "web",
    }

    # 文档标注为"一般情况可以重试解决"的错误码：
    # 10500 InnerError(默认内部错误)、700429 并发量超过 QPS 限流。
    # 其余(10400 参数 / 10402 搜索类型 / 10403 权限 / 10406 免费额度用尽 /
    # 10409·10410·10412 套餐问题)都需要人工介入，重试无益。
    RETRYABLE_ERROR_CODES = {"10500", "700429"}

    def _do_search(
        self,
        client: httpx.Client,
        query: str,
        options: SearchOptions,
    ) -> list[dict[str, Any]]:
        # 1. 映射搜索类型，并按类型钳制 Count 上限
        doubao_type = self.SEARCH_TYPE_MAP.get(options.search_type, "web")
        max_count = self.MAX_COUNT_IMAGE if doubao_type == "image" else self.MAX_COUNT_WEB
        count = max(1, min(options.num_results, max_count))

        # 2. 组 Filter：站点白/黑名单按文档上限截断，避免整请求被判 10400
        filters: dict[str, Any] = {
            # 恒为 True：本工具的产出必须带可点击链接，
            # 代价是会滤掉没有落地页的"火山如意"卡片结果(我们也用不到)
            "NeedUrl": True,
            "NeedContent": options.need_content,
        }
        sites = options.split_sites()[: self.MAX_SITES]
        if sites:
            filters["Sites"] = "|".join(sites)
        block_hosts = options.split_block_hosts()[: self.MAX_BLOCK_HOSTS]
        if block_hosts:
            filters["BlockHosts"] = "|".join(block_hosts)
        if options.authority_only:
            filters["AuthInfoLevel"] = 1  # 1 = 仅"非常权威"
        if options.industry:
            filters["Industry"] = options.industry

        # 3. 组请求体；Query 超长先截断，避免上游静默截断导致语义漂移
        payload: dict[str, Any] = {
            "Query": query[: self.MAX_QUERY_CHARS],
            "SearchType": doubao_type,
            "Count": count,
            "Filter": filters,
        }
        if options.time_range:
            payload["TimeRange"] = options.time_range
        if options.query_rewrite:
            payload["QueryControl"] = {"QueryRewrite": True}
        if options.full_content:
            payload["ContentFormats"] = options.content_format

        # 4. 发起请求
        response = client.post(
            f"{self.base_url}/search_api/web_search",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # 5. 关键：业务错误是在 HTTP 200 里通过 ResponseMetadata.Error 返回的，
        #    只看 HTTP 状态码会把失败当成功
        metadata: dict[str, Any] = data.get("ResponseMetadata") or {}
        error: dict[str, Any] | None = metadata.get("Error")
        if error:
            code_n = error.get("CodeN")
            code = str(error.get("Code") or code_n or "unknown")
            detail = f"豆包搜索业务错误 [{code}] {error.get('Message', '')}"
            # 按文档区分可重试(服务端内部错误 / 限流)与不可重试(账号·套餐·参数)
            if {code, str(code_n)} & self.RETRYABLE_ERROR_CODES:
                raise RetryableUpstreamError(detail)
            raise SearchProviderError(detail)

        result: dict[str, Any] = data.get("Result") or {}

        # 6. 图片搜索走 ImageResults，其余走 WebResults
        if doubao_type == "image":
            return [
                {
                    "title": item.get("Title") or "Untitled",
                    "link": item.get("Url", ""),
                    "snippet": item.get("Category") or "",
                    "date": _clean_date(item.get("PublishTime")),
                    "source": item.get("SiteName"),
                }
                for item in (cast(list[dict[str, Any]], result.get("ImageResults") or []))[:count]
            ]

        # 7. 归一 WebResults：
        #    full_content=True 取 Content(全文)，否则取 Summary(长摘要)，
        #    Snippet(约 200 字，官方不建议喂模型)只作最后兜底
        results: list[dict[str, Any]] = []
        web_results: list[dict[str, Any]] = result.get("WebResults") or []
        for item in web_results[:count]:
            if options.full_content:
                snippet = item.get("Content") or item.get("Summary") or item.get("Snippet") or ""
            else:
                snippet = item.get("Summary") or item.get("Snippet") or ""
            results.append(
                {
                    "title": item.get("Title") or "Untitled",
                    "link": item.get("Url", ""),
                    "snippet": snippet,
                    "date": _clean_date(item.get("PublishTime")),
                    "source": item.get("SiteName"),
                    "position": item.get("SortId"),
                    "authority": item.get("AuthInfoDes"),
                }
            )
        return results


class BaiduProvider(SearchProviderBase):
    """Baidu(百度 AI 搜索 / 千帆)——`POST /v2/ai_search/web_search`，返回 references。

    官方文档:
        - 「百度搜索」API 参考(本类实现依据，含 search_filter 完整结构):
          https://cloud.baidu.com/doc/qianfan-api/s/Wmbq4z7e5
        - API Key 说明: https://cloud.baidu.com/doc/BAIDU_AI_SEARCH/s/5mkmgi38d
        - 控制台(建 API Key): https://console.bce.baidu.com/ai-search/home

    ⚠️ 排错提示：**401 不能证明路径正确**——实测网关在 `/v2/ai_search` 前缀上先鉴权
    再匹配子路径，连 `/v2/ai_search/definitely_not_exist` 都返回 401。因此换 base_url
    或改路径后，要用真实 key 打一次成功请求来确认，不能只看"没报 404"。
    """

    name = "Baidu"
    default_base_url = "https://qianfan.baidubce.com"
    # full_content 在此为 no-op(snippet 与 content 返回同一段文本)，也一并提示
    IGNORED_PARAMS = frozenset(
        {
            "authority_only",
            "industry",
            "query_rewrite",
            "need_content",
            "full_content",
            "content_format",
            "country",
            "language",
            "location",
            "page",
        }
    )

    # 文档约束：messages[].content 上限 72 字符
    MAX_QUERY_CHARS = 72

    # time_range 预设 -> `search_recency_filter`。
    # 文档合法值只有 week / month / semiyear / year，**没有 day**——
    # 因此 OneDay 不走这里，改用 `search_filter.range.page_time` 精确到日。
    RECENCY_FOR_TIME_RANGE = {
        "OneWeek": "week",
        "OneMonth": "month",
        "OneYear": "year",
    }

    def _build_search_filter(self, options: SearchOptions) -> dict[str, Any]:
        """组装 `search_filter`：站点白/黑名单与发文日期区间。

        结构取自官方文档：
            search_filter.match.site        -> array[str]  限定站点
            search_filter.block_websites    -> array[str]  屏蔽站点
            search_filter.range.page_time   -> {gte, lte}  发文时间区间
        """
        search_filter: dict[str, Any] = {}

        sites = options.split_sites()
        if sites:
            search_filter["match"] = {"site": sites}
        blocked = options.split_block_hosts()
        if blocked:
            # 仍然发送：实测上游**不生效**(完整域名与裸域名都试过，结果不变)，
            # 但字段是文档明列的，留着以便上游日后修好；真正的屏蔽靠下面的本地过滤
            search_filter["block_websites"] = blocked

        # 自定义区间直接落到 page_time；OneDay 因为上游没有 day 枚举，
        # 也退到这里用"今天起"表达
        page_time: dict[str, str] = {}
        custom = options.parse_custom_range()
        if custom:
            page_time["gte"], page_time["lte"] = custom
        elif options.time_range == "OneDay":
            today = datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
            page_time["gte"] = today
        if page_time:
            search_filter["range"] = {"page_time": page_time}

        return search_filter

    def _do_search(
        self,
        client: httpx.Client,
        query: str,
        options: SearchOptions,
    ) -> list[dict[str, Any]]:
        # 1. 组请求体；query 超长先截断，避免上游静默截断导致语义漂移
        # 屏蔽名单靠本地过滤实现，先多要一些结果留出被滤掉的余量
        blocked = options.split_block_hosts()
        top_k = min(options.num_results * 2, 50) if blocked else options.num_results
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": query[: self.MAX_QUERY_CHARS]}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": top_k}],
        }

        # 2. 时效：预设枚举走 search_recency_filter，
        #    自定义区间与 OneDay 走 search_filter.range.page_time
        recency = self.RECENCY_FOR_TIME_RANGE.get(options.time_range)
        if recency:
            payload["search_recency_filter"] = recency

        # 3. 站点过滤与日期区间
        search_filter = self._build_search_filter(options)
        if search_filter:
            payload["search_filter"] = search_filter

        # 4. 发起请求
        response = client.post(
            f"{self.base_url}/v2/ai_search/web_search",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        # 5. 业务错误：文档标注失败时带 code / message
        if data.get("code") and not data.get("references"):
            raise SearchProviderError(f"百度 AI 搜索业务错误 [{data.get('code')}] {data.get('message', '')}")

        # 6. 归一 references。
        #    实测 `snippet` 与 `content` 返回的是**同一段文本**(逐字相同)，
        #    所以 `full_content` 在本服务商上是 no-op；两个都取只是防上游日后分化。
        lowered = [b.lower() for b in blocked]
        results: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = data.get("references") or []
        for ref in references:
            if len(results) >= options.num_results:
                break
            # 本地兜底屏蔽(上游 block_websites 无效)
            if lowered and any(_host_matches(_host_of(ref.get("url") or ""), b) for b in lowered):
                continue
            if options.full_content:
                snippet = ref.get("content") or ref.get("snippet") or ""
            else:
                snippet = ref.get("snippet") or ref.get("content") or ""
            results.append(
                {
                    "title": ref.get("title") or "Untitled",
                    "link": ref.get("url", ""),
                    "snippet": snippet,
                    "date": _clean_date(ref.get("date")),
                    # 实测 `web_anchor` 常为空串、`website` 偶尔是字面量 "无"，
                    # 两个都当缺失处理，免得把 source="无" 喂给模型
                    "source": _clean_source(ref.get("web_anchor"), ref.get("website")),
                }
            )
        return results


class XiaoBeiProvider(SearchProviderBase):
    """XiaoBei(北坡 Search Engine Orchestrator)——多引擎聚合 + 全页面抓取。

    上游: `POST https://search.xiaobei.top/tools/web_research`(内网 `http://search.iqjzf.com`)
    鉴权: `X-API-Key: <API_KEY>`

    官方文档:
        - 内置文档页(端点速查 / 参数表 / 状态机 / 错误码): https://search.xiaobei.top/docs
        - OpenAPI schema: https://search.xiaobei.top/openapi.json
        - 健康检查(看 redis / searxng / firecrawl 依赖是否正常): https://search.xiaobei.top/health

    限流(摘自文档): 搜索每 Key 300 req/min、全局 500 req/min；
    采集每 Key 300 req/min、全局 600 req/min。

    它自己就是个聚合层：一次请求并发打 google / bing / baidu，再按相关性融合排序，
    可选地用 Firecrawl 抓取前 N 条的正文。因此有几处与单一引擎很不一样的地方：

    1. **`max_results` 是"每引擎"量级而非总量**，实测传 5 会回 14 条、传 30 回 27 条，
       所以本地必须再按 `num_results` 截断一次。
    2. **上游会抖**：单个引擎 `timeout` / `empty_results` 很常见，整体也会 502
       (`searxng timeout`)。"HTTP 200 + 0 条结果 + 有 engine_errors"是**临时故障**
       而非"搜不到"，必须重试，否则会把抖动当成空结果返给模型。
    3. **`status="failed"` 指的是"所有*抓取*失败"，不是搜索失败**(那是 `search_failed`)。
       只要 `results` 有内容就仍然可用，不能当致命错误。
    4. 单轮搜索就要 15~25 秒，全局 `SEARCH_TIMEOUT`(默认 30s)对它过紧，
       这里用引擎自己的预算下限兜底。
    """

    name = "XiaoBei"
    default_base_url = "https://search.xiaobei.top"
    IGNORED_PARAMS = frozenset(
        {
            "authority_only",
            "industry",
            "query_rewrite",
            "need_content",
            "content_format",
            "country",
            "location",
            "page",
        }
    )
    supports_engine_choice = True
    supports_render_js = True

    # 文档约束：max_results 1~30、scrape_top_n 0~10、max_content_chars 100~50000
    MAX_RESULTS_CAP = 30
    MAX_SCRAPE_TOP_N = 10
    MAX_CONTENT_CHARS_CAP = 50000
    # 可选的下游搜索引擎
    SUPPORTED_ENGINES = ("google", "bing", "baidu")

    # 单次 HTTP 请求的上限。上游前面挂着网关，**约 180s 就会切 504**，
    # 所以任何单个请求都必须压在这条线以下——抓正文改走异步轮询正是为此。
    # 搜索本身实测 15~40s，100s 留足余量。
    HTTP_TIMEOUT_CEILING = 100
    # 上游 timeout_s 比本地 HTTP 超时小这么多，留给网络与序列化
    BUDGET_MARGIN = 25
    # 抓正文的轮询预算(秒)。任务进终态就提前退出，不会真等满。
    # 抓取正常时 Firecrawl 一般 2~30s 就落定；给 60s 足够，
    # 抓取整体故障时也能快点降级回搜索摘要，不把工具调用拖到几分钟。
    SCRAPE_POLL_BUDGET = 60
    # 轮询间隔
    POLL_INTERVAL = 8

    # 本工具的 time_range 预设 -> 上游枚举
    TIME_RANGE_MAP = {
        "OneDay": "day",
        "OneWeek": "week",
        "OneMonth": "month",
        "OneYear": "year",
    }

    # 鉴权 / 参数类错误，重试无用
    NON_RETRYABLE_STATUS = {401, 403, 422}
    # 已抵达终态、不必再轮询的任务状态
    TERMINAL_STATUSES = {"complete", "partial", "failed", "search_failed"}

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    @staticmethod
    def _parse_json(response: httpx.Response) -> dict[str, Any]:
        """上游偶发返回空体/非 JSON，单独兜一层给出可读错误而不是 JSONDecodeError。"""
        try:
            return response.json()
        except ValueError as exc:
            raise RetryableUpstreamError(f"上游返回非 JSON 响应(HTTP {response.status_code}): {response.text[:120]!r}") from exc

    def _check_status(self, response: httpx.Response) -> None:
        """把 401/403/422 归到不可重试的配置错误，其余交给基类统一处理。"""
        if response.status_code in self.NON_RETRYABLE_STATUS:
            detail = response.text[:200]
            raise SearchProviderError(f"北坡搜索拒绝请求 HTTP {response.status_code}: {detail}")
        response.raise_for_status()

    def _do_search(
        self,
        client: httpx.Client,
        query: str,
        options: SearchOptions,
    ) -> list[dict[str, Any]]:
        # 1. 是否需要抓正文。
        #    只搜不抓 -> 同步一发拿结果，最简单；
        #    要抓正文 -> **异步**：首响应等搜索完成(~15~40s)就带回结果，
        #    抓取部分再轮询。这样任何单个请求都不会逼近网关的 180s 切断线。
        scrape_top_n = min(options.num_results, self.MAX_SCRAPE_TOP_N) if options.full_content else 0
        http_timeout = float(min(max(self.timeout, 60.0), self.HTTP_TIMEOUT_CEILING))
        upstream_timeout = max(5, min(600, int(http_timeout) - self.BUDGET_MARGIN))

        # 2. 上游没有结构化站点过滤字段，但它把 query 透传给 google/bing/baidu，
        #    因此先用 `site:` / `-site:` 算子在上游侧收敛(实测 1/28 → 20/20)，
        #    再在本地按 host 兜底过滤——个别下游引擎会忽略算子。
        sites = options.split_sites()
        block_hosts = options.split_block_hosts()
        effective_query = _apply_site_operators(query, sites, block_hosts)

        # 开启过滤时多要一些结果，给本地兜底过滤留余量
        want_filter = bool(sites or block_hosts)
        max_results = self.MAX_RESULTS_CAP if want_filter else options.num_results
        max_results = max(1, min(self.MAX_RESULTS_CAP, max_results))

        upstream_options: dict[str, Any] = {
            "max_results": max_results,
            "scrape_top_n": scrape_top_n,
            "max_content_chars": min(options.max_content_chars, self.MAX_CONTENT_CHARS_CAP),
            "wait_for_result": not scrape_top_n,
            "timeout_s": upstream_timeout,
            # 上游是双负逻辑：fast_mode=False 才是浏览器渲染
            "fast_mode": not options.render_js,
        }
        # 3. 逐个映射可选参数
        requested = options.split_engines()
        engines = [e for e in requested if e in self.SUPPORTED_ENGINES]
        for unknown in set(requested) - set(engines):
            logger.warning("北坡搜索不支持引擎 %r，已忽略(可选 %s)", unknown, self.SUPPORTED_ENGINES)
        if engines:
            upstream_options["engines"] = engines
        if options.language:
            upstream_options["language"] = options.language
        mapped_range = self.TIME_RANGE_MAP.get(options.time_range)
        if mapped_range:
            upstream_options["time_range"] = mapped_range
        elif options.time_range:
            # 上游只认 day/week/month/year，自定义区间不支持，忽略但留痕
            logger.warning("北坡搜索不支持 time_range=%r，已忽略", options.time_range)

        # 4. 发起请求(用引擎自己的预算覆盖 client 全局超时)
        response = client.post(
            f"{self.base_url}/tools/web_research",
            headers=self._headers(),
            json={"query": effective_query, "options": upstream_options},
            timeout=http_timeout,
        )
        self._check_status(response)
        data = self._parse_json(response)

        # 5. 需要正文但抓取还没结束时，按预算轮询任务
        if scrape_top_n and not self._scrape_settled(data):
            data = self._poll_task(client, data, self.SCRAPE_POLL_BUDGET)

        return self._normalize(data, options)

    @staticmethod
    def _scrape_settled(data: dict[str, Any]) -> bool:
        """抓取是否已尘埃落定(任务终态，或每条 scrape 都不再变化)。"""
        if data.get("status") in XiaoBeiProvider.TERMINAL_STATUSES:
            return True
        pending = {"queued", "scraping", "pending"}
        items: list[dict[str, Any]] = data.get("results") or []
        for item in items:
            scrape: dict[str, Any] = item.get("scrape") or {}
            if scrape.get("status") in pending:
                return False
        return True

    def _poll_task(
        self,
        client: httpx.Client,
        data: dict[str, Any],
        budget: int,
    ) -> dict[str, Any]:
        """轮询 `GET /tasks/{id}` 直到抓取落定或预算耗尽。

        拿不到更好的结果就退回上一次的响应——搜索结果本身已经可用，
        不能因为抓正文没成功就把整次检索判失败。
        """
        task_id = data.get("task_id")
        if not task_id:
            return data
        deadline = time.monotonic() + budget
        latest = data
        while time.monotonic() < deadline:
            time.sleep(self.POLL_INTERVAL)
            try:
                resp = client.get(
                    f"{self.base_url}/tasks/{task_id}",
                    headers=self._headers(),
                    timeout=30.0,
                )
                self._check_status(resp)
                latest = self._parse_json(resp)
            except SearchProviderError:
                raise
            except Exception as exc:  # noqa: BLE001 — 轮询失败不该毁掉已有结果
                logger.warning("轮询任务 %s 失败: %s", task_id, exc)
                return latest
            if self._scrape_settled(latest):
                break
        return latest

    def _normalize(
        self,
        data: dict[str, Any],
        options: SearchOptions,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = data.get("results") or []
        engine_errors: list[dict[str, Any]] = data.get("engine_errors") or []
        failure_reason = data.get("failure_reason")
        status = data.get("status")

        # 6. 搜索本身失败 → 明确报错(注意 status="failed" 只代表抓取失败，不在此列)
        if status == "search_failed" or failure_reason == "all_engines_error":
            raise RetryableUpstreamError(f"北坡搜索失败 status={status} failure_reason={failure_reason} engine_errors={engine_errors}")

        # 7. 空结果的两种可能：上游抖动(有 engine_errors，值得重试) vs 真没有(直接返空)
        if not results:
            if engine_errors:
                raise RetryableUpstreamError(f"北坡搜索本轮无结果，下游引擎报错 {engine_errors}，重试")
            return []

        # 8. 站点白/黑名单：上游没有该能力，本地按 host 过滤
        sites = [s.lower() for s in options.split_sites()]
        blocked = [b.lower() for b in options.split_block_hosts()]
        normalized: list[dict[str, Any]] = []
        for item in results:
            url = item.get("url") or ""
            host = _host_of(url)
            if sites and not any(_host_matches(host, s) for s in sites):
                continue
            if blocked and any(_host_matches(host, b) for b in blocked):
                continue

            # 抓到正文就优先用正文，否则回落到搜索摘要
            scrape: dict[str, Any] = item.get("scrape") or {}
            markdown: str = scrape.get("markdown") or ""
            snippet: str = markdown if (options.full_content and markdown) else (item.get("snippet") or "")

            normalized.append(
                {
                    "title": item.get("title") or "Untitled",
                    "link": url,
                    "snippet": snippet,
                    "date": _clean_date(item.get("published_date")),
                    # 聚合服务商特有：这条结果实际由哪个底层引擎搜到，溯源用
                    "engine": item.get("engine"),
                    "position": item.get("rank"),
                }
            )
        return normalized


# 引擎注册表：环境变量值(小写)-> 适配器类
_PROVIDER_REGISTRY: dict[str, type[SearchProviderBase]] = {
    "serper": SerperProvider,
    "seed": SeedProvider,
    "baidu": BaiduProvider,
    "xiaobei": XiaoBeiProvider,
}

# 引擎名别名，容忍中文写法与服务自称
_PROVIDER_ALIASES = {
    "小北": "xiaobei",
    "orchestrator": "xiaobei",
    "北坡": "xiaobei",
    "doubao": "seed",
    "豆包": "seed",
    "百度": "baidu",
}

# 引擎实例缓存，key 为配置签名——配置变了自动重建，避免缓存住旧凭证
_provider_cache: dict[tuple[str, ...], SearchProviderBase] = {}


def _resolve_provider_name() -> str:
    """读取 `SEARCH_PROVIDER`——**服务商只认这一个变量名**，缺省 Serper。"""
    return (os.getenv("SEARCH_PROVIDER") or "").strip() or "Serper"


# 向后兼容：本 RFC 之前内置搜索只认 SERPER_API_KEY。
# 老用户不改任何配置也应继续可用，因此 SEARCH_API_KEY 缺省时回落到它。
_LEGACY_API_KEY_ENV = "SERPER_API_KEY"


def _resolve_api_key() -> str:
    """读取 `SEARCH_API_KEY`；缺省时回落 `SERPER_API_KEY`(向后兼容)。"""
    value = (os.getenv("SEARCH_API_KEY") or "").strip()
    if value:
        return value
    return (os.getenv(_LEGACY_API_KEY_ENV) or "").strip()


def _get_provider() -> SearchProviderBase:
    """按当前环境变量拿到(或构建)引擎实例。"""
    # 1. 解析引擎名并查注册表
    raw_name = _resolve_provider_name()
    provider_key = raw_name.strip().lower()
    provider_key = _PROVIDER_ALIASES.get(provider_key, provider_key)
    provider_cls = _PROVIDER_REGISTRY.get(provider_key)
    if provider_cls is None:
        raise SearchProviderError(f"Unsupported SEARCH_PROVIDER: {raw_name!r}. Supported providers: Serper / Seed / Baidu / XiaoBei.")

    # 2. 解析密钥与可选覆盖项
    api_key = _resolve_api_key()
    base_url = (os.getenv("SEARCH_BASE_URL") or "").strip() or None
    timeout = float(os.getenv("SEARCH_TIMEOUT") or DEFAULT_TIMEOUT)
    max_retries = int(os.getenv("SEARCH_MAX_RETRIES") or DEFAULT_MAX_RETRIES)

    # 3. 配置签名命中缓存则复用
    signature = (provider_key, api_key, base_url or "", str(timeout), str(max_retries))
    cached = _provider_cache.get(signature)
    if cached is not None:
        return cached

    # 4. 构建实例
    engine = provider_cls(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    _provider_cache[signature] = engine
    return engine


def _clean_source(*candidates: Any) -> str | None:
    """按顺序取第一个有意义的来源名。

    百度侧实测 `web_anchor` 常为空串、`website` 偶尔返回字面量 "无"——
    都视作缺失，避免把 `source="无"` 当真来源喂给模型。
    """
    junk = {"", "无", "未知", "null", "none"}
    for value in candidates:
        if isinstance(value, str) and value.strip().lower() not in junk:
            return value.strip()
    return None


def _clean_date(value: Any) -> str | None:
    """归一发布时间：上游对没有时间的结果(如图片)会回 1970 epoch，视作缺失。"""
    if not value or not isinstance(value, str):
        return None
    return None if value.startswith("1970") else value


def _results_to_llm_content(results: list[dict[str, Any]]) -> str:
    """把结果列表格式化成 gemini-cli 风格文本(与内置 WebSearch 一致)。"""
    lines: list[str] = []
    for idx, item in enumerate(results, 1):
        title = item.get("title", "Untitled")
        link = item.get("link") or item.get("url") or "No URL"
        lines.append(f"[{idx}] {title} ({link})")
        if item.get("snippet"):
            lines.append(f"    {item['snippet']}")
        meta = [
            f"date: {item['date']}" if item.get("date") else "",
            f"authority: {item['authority']}" if item.get("authority") else "",
        ]
        meta_line = ", ".join(m for m in meta if m)
        if meta_line:
            lines.append(f"    ({meta_line})")
    return "\n".join(lines) if lines else "No results found."


def _error_result(message: str, error_type: str = "WEB_SEARCH_FAILED") -> dict[str, Any]:
    """统一的错误返回结构，与内置 `google_web_search` 对齐。"""
    return {
        "content": f"Error: {message}",
        "returnDisplay": "Error performing web search.",
        "error": {"message": message, "type": error_type},
    }


def web_search(
    query: str,
    num_results: int | None = None,
    search_type: str | None = None,
    time_range: str | None = None,
    sites: str | None = None,
    block_hosts: str | None = None,
    authority_only: bool | None = None,
    industry: str | None = None,
    query_rewrite: bool | None = None,
    need_content: bool | None = None,
    full_content: bool | None = None,
    content_format: str | None = None,
    max_content_chars: int | None = None,
    country: str | None = None,
    language: str | None = None,
    location: str | None = None,
    page: int | None = None,
    search_engine: str | None = None,
    render_js: bool | None = None,
) -> dict[str, Any]:
    """按 `SEARCH_PROVIDER` 指定的服务商执行 Web 搜索。

    除 `query` 外全部参数都可省略；**全部省略时，行为与 Nexau 内置 WebSearch 等价**。
    服务商不支持的参数会被静默忽略(见模块 docstring 的支持度矩阵)。

    **取值优先级：调用方显式传参 > 环境变量 > 内置默认值。**
    每个参数都有对应的环境变量供部署方设"全局默认"，命名规则为
    `SEARCH_` + 参数名大写(自带 `search_` 前缀的去重)，例如
    `content_format` → `SEARCH_CONTENT_FORMAT`、`search_engine` → `SEARCH_ENGINE`。

    签名里的默认值一律是 `None` 哨兵，只有 `None` 才表示"没传"——
    这样才能区分"没传"与"显式传了与默认值相同的值"(实测模型会原样回传默认值，
    若按值判等会让部署方设的默认时灵时不灵)。内置默认值见 `PARAM_DEFAULTS`，
    与 tool YAML 的 `default` 保持同步。

    Args:
        query: 搜索关键词。豆包侧限 1~100 字符，超长会截断；不支持多词并列检索。
        num_results: 返回结果条数，默认 10。上限按引擎/类型钳制(豆包 web≤50、image≤5)。
        search_type: 搜索类型，默认 `search`。可选 `search` 网页 / `news` 资讯 /
            `images` 图片 / `places` 地点 / `videos` 视频 / `scholar` 学术论文。
            Serper 六类均为独立端点；豆包只有 web 与 image 两种原生类型，
            `news` 会自动附加"最近一周"时效，其余退化为网页搜索。
        time_range: 时效过滤，默认 `""`(不限制)。可选 `OneDay` / `OneWeek` /
            `OneMonth` / `OneYear`，或 `YYYY-MM-DD..YYYY-MM-DD` 自定义区间
            (Serper 编译成 `tbs=cdr:…`，豆包原生支持)。显式传值会覆盖 `news` 的默认一周。
        sites: 限定站点，默认 `""`(不限)。多个用 `|` 分隔完整域名，最多 20 个，
            如 `arxiv.org|nature.com`。
        block_hosts: 屏蔽站点，默认 `""`(不屏蔽)。多个用 `|` 分隔，最多 5 个。
        authority_only: 是否只要"非常权威"来源(政府/央媒/国家机构等)，默认 False。
            开启后结果数会明显减少。仅豆包支持。
        industry: 行业垂直检索，默认 `""`(不限)。可选 `finance` 金融 /
            `game` 电子游戏 / `gov` 政务。仅豆包支持。
        query_rewrite: 是否让上游改写查询词以提升召回，默认 False(会增加耗时)。
            Serper 上对应 Google 拼写纠正 `autocorrect`；豆包上对应 `QueryRewrite`。
        need_content: 是否只保留"成功抓到正文"的结果，默认 False。用于过滤空壳页。
        full_content: 是否用**完整正文**替代摘要，默认 False。开启后单条内容极长，
            务必配合 `max_content_chars` 使用。
        content_format: 正文格式，默认 `text`，可选 `markdown`。仅 `full_content=True` 时有意义。
        max_content_chars: 单条摘要/正文的截断长度，默认 1000 字符。
        country: 检索区域，ISO 3166-1 两位国家码(如 `cn` / `us`)，默认 `""`(跟随上游)。
            仅 Serper 支持(映射 `gl`)。
        language: 结果语言(如 `zh-cn` / `en`)，默认 `""`(跟随上游)。
            仅 Serper 支持(映射 `hl`)。
        location: 地理定位描述串(如 `"Shanghai, China"`)，默认 `""`(不定位)。
            做本地生活类检索或 `search_type=places` 时建议填。仅 Serper 支持。
        page: 结果页码，从 1 开始，默认 1。用于翻页取更靠后的结果。仅 Serper 支持。
        search_engine: 聚合引擎的下游搜索引擎子集，`|` 分隔，如 `"google|baidu"`，
            默认 `""`(用上游默认的 google+bing+baidu)。中文内容偏好 baidu，
            英文/技术内容偏好 google。仅 XiaoBei 支持。
        render_js: 抓正文时是否用浏览器渲染，默认 False(走 HTTP 快速引擎)。
            开启后能拿到 JS 动态生成的内容，代价是明显更慢。
            **只在 `full_content=True` 时有意义**——不抓正文就没有渲染这一步。
            仅 XiaoBei 支持(映射到上游 `fast_mode` 取反)。

    Returns:
        gemini-cli 风格 dict：成功含 `content` / `returnDisplay` / `sources` / `provider`，
        失败含 `content` / `returnDisplay` / `error`。**任何情况下都不抛异常。**

        注意顶层是 `provider`(服务商名)；`engine`(底层搜索引擎)只出现在 `sources`
        的单条结果里，且仅聚合型服务商会写。两者的区别见模块 docstring「术语」一节。
    """
    try:
        # 1. 入参校验(与内置一致，空 query 直接短路)
        if not query or not query.strip():
            return {
                "content": "The 'query' parameter cannot be empty.",
                "returnDisplay": "Error: Empty search query.",
                "error": {
                    "message": "The 'query' parameter cannot be empty.",
                    "type": "INVALID_QUERY",
                },
            }

        # 2. 逐参数按「入参 > 环境变量 > 内置默认」解析，再收敛成 options
        #    （__post_init__ 里做 news→时效 等派生逻辑）
        supplied = {
            "num_results": num_results,
            "search_type": search_type,
            "time_range": time_range,
            "sites": sites,
            "block_hosts": block_hosts,
            "authority_only": authority_only,
            "industry": industry,
            "query_rewrite": query_rewrite,
            "need_content": need_content,
            "full_content": full_content,
            "content_format": content_format,
            "max_content_chars": max_content_chars,
            "country": country,
            "language": language,
            "location": location,
            "page": page,
            "search_engine": search_engine,
            "render_js": render_js,
        }
        options = SearchOptions(**{name: resolve_param(name, value) for name, value in supplied.items()})

        # 3. 按环境变量拿引擎并执行(引擎层已做重试；失败返回错误字符串)
        provider = _get_provider()
        results = provider.search(query=query, options=options)
        if isinstance(results, str):
            return _error_result(f"[{provider.name}] {results}")

        if not results:
            return {
                "content": f'Web search results for "{query}":\n\nNo results found.',
                "returnDisplay": f'Search results for "{query}" returned (0 results).',
                "sources": [],
                "provider": provider.name,
            }

        # 4. 格式化成 gemini-cli 风格文本
        formatted = _results_to_llm_content(results)
        return {
            "content": f'Web search results for "{query}" (provider: {provider.name}):\n\n{formatted}',
            "returnDisplay": (f'Search results for "{query}" via {provider.name} returned ({len(results)} results).'),
            "sources": results,
            "provider": provider.name,
        }

    except SearchProviderError as e:
        # 配置类错误(缺 key / 引擎名非法 / 账号额度)，单独标注类型便于排查
        return _error_result(str(e), error_type="WEB_SEARCH_CONFIG_ERROR")

    except Exception as e:  # noqa: BLE001 — 工具层兜底，绝不把异常抛回 Runtime
        logger.exception("web_search failed")
        return _error_result(f"{type(e).__name__}: {str(e)}")
