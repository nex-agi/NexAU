# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportReturnType=false
# pyright: reportAssignmentType=false
# pyright: reportCallIssue=false
# pyright: reportOptionalCall=false
# pyright: reportAttributeAccessIssue=false

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

import asyncio
import json
import logging
import os
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Annotated, Any, TypeVar, cast

import lark_oapi as lark  # type: ignore[import-untyped]
import requests
from lark_oapi.api.im.v1 import (  # type: ignore[import-untyped]
    CreateFileRequest,
    CreateFileRequestBody,
    CreateFileResponse,
    CreateImageRequest,
    CreateImageRequestBody,
    CreateImageResponse,
    CreateMessageReactionRequest,
    CreateMessageReactionRequestBody,
    CreateMessageReactionResponse,
    CreateMessageRequest,
    CreateMessageRequestBody,
    CreateMessageResponse,
    Emoji,
    GetChatMembersRequest,
    GetChatMembersResponse,
    ListChatRequest,
    ListChatResponse,
    ListMessageRequest,
    ListMessageResponse,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
    ReplyMessageResponse,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
JSONDict = dict[str, Any]


# Runtime validation helpers
def _ensure_dict(value: object, *, context: str) -> JSONDict:
    """Validate that a value is a dictionary and return it."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a dict, got {type(value).__name__}")
    return cast(JSONDict, value)


def _ensure_list_of_dicts(value: object, *, context: str) -> list[JSONDict]:
    """Validate that a value is a list of dictionaries and return it."""
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list, got {type(value).__name__}")
    value_list = cast(list[object], value)
    dict_items: list[JSONDict] = []
    for index, item in enumerate(value_list):
        if not isinstance(item, dict):
            raise ValueError(f"{context}[{index}] must be a dict, got {type(item).__name__}")
        dict_items.append(cast(JSONDict, item))
    return dict_items


# Helper to marshal Lark JSON with a stable return type
def _marshal_json(data: Any, indent: int | None = 4) -> str:
    json_marshal = cast(Callable[[Any, int | None], str | None], getattr(lark.JSON, "marshal", None))
    result = json_marshal(data, indent) if json_marshal is not None else None
    return result or ""


# Helper to safely extract the first string from a sequence of values
def _first_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            return value
    return ""


# Helper to ensure the Lark client is treated as dynamically typed for pyright
def _create_lark_client(app_id: str, app_secret: str) -> Any:
    return cast(
        Any,
        lark.Client.builder().app_id(app_id).app_secret(app_secret).log_level(lark.LogLevel.INFO).build(),
    )


"""
飞书机器人工具集合 - 精简版

本模块提供了核心的飞书API工具函数，专为大模型调用优化，工具少而精：

🔍 消息查询工具：
   - get_feishu_chat_list: 获取用户或机器人所在的群列表
   - get_feishu_message_list: 获取指定会话内的历史消息

📤 消息发送工具：
   - send_feishu_message: 统一的消息发送函数，支持所有消息类型（文本/富文本/图片/卡片）
   - reply_to_feishu_message: 回复指定消息
   - add_message_reaction: 给消息添加表情回应

🔍 用户搜索工具：
   - search_users_in_chat: 获取指定群组中所有用户的信息列表
   - get_user_info_by_id: 根据用户ID获取用户详细信息
   - get_user_id_by_name: 根据用户名查找用户信息，支持重名检测

📎 辅助工具：
   - upload_feishu_image: 上传图片文件，获取image_key用于发送图片消息（支持沙盒文件）
   - upload_feishu_file: 上传文件，获取file_key用于发送文件消息（支持沙盒文件）

核心设计理念：
✅ 工具函数少而精，减少大模型选择困难
✅ 详细的使用文档内嵌在函数中
✅ 统一的错误处理和参数格式
✅ 支持所有常用消息类型
✅ 智能重名检测，避免误发消息

沙盒文件支持：
- upload_feishu_image 和 upload_feishu_file 自动检测沙盒中的文件
- 如果文件路径在当前沙盒中存在，会自动从沙盒下载并上传

使用说明：
- 所有函数从运行时配置、环境变量或默认值获取飞书APP凭证（优先级递减）
- 配置方式：通过config["configurable"]["secrets"]["feishu"]传递app_id和app_secret
- receive_id_type: chat_id(群组)/open_id(用户)/user_id/union_id/email
- 消息类型: text(文本)/post(富文本)/image(图片)/interactive(卡片)
- 查找用户建议先使用 get_user_id_by_name 获取准确的用户信息，再发送消息
"""


# 飞书配置获取函数
def get_feishu_config():
    """
    获取飞书配置

    优先级：
    1. 运行时配置（通过 .secrets.user.json）
    2. 直接读取.secrets.user.json文件
    3. 环境变量

    配置方法：
    - Step 1: 创建 .secrets.user.json 文件，包含 {"feishu": {"app_id": "xxx", "app_secret": "xxx"}}
    - Step 2: 在 .env 中配置 SECRETS_USER_PATH=/path/to/your/.secrets.user.json
    - Step 3: 配置会自动注入到每个node的config中

    返回：
    - 成功：(app_id, app_secret) 元组
    - 失败：None（表示配置缺失或无效）
    """
    try:
        # 3. 尝试从环境变量获取
        app_id = os.getenv("FEISHU_APP_ID")
        app_secret = os.getenv("FEISHU_APP_SECRET")
        if app_id and app_secret:
            logger.info("从环境变量获取飞书配置")
            return app_id, app_secret

        # 4. 所有配置获取方法都失败
        logger.error("未找到飞书配置")
        return None

    except Exception as e:
        logger.error(f"获取飞书配置异常: {e}")
        return None


def validate_feishu_config():
    """
    验证飞书配置并返回结果

    返回：
    - 成功：(app_id, app_secret) 元组
    - 失败：错误描述字符串
    """
    config_result = get_feishu_config()

    if config_result is None:
        error_msg = """飞书配置未设置，无法使用飞书工具。"""

        return error_msg

    return config_result


def get_user_id_by_feishu_id(feishu_id: str) -> str | None:
    """
    通过飞书ID获取系统用户ID

    参数：
    - feishu_id: 飞书用户ID（open_id等）

    返回：
    - 成功：系统用户ID字符串
    - 失败：None
    """
    try:
        # 获取API基础URL，默认为本地环境
        xiaobei_api_base_url = os.getenv(
            "XIAOBEI_API_BASE_URL",
            "http://localhost:8000",
        )

        url = f"{xiaobei_api_base_url}/api/v1/admin/get-user-by-feishu-id"
        data = {
            "feishu_id": feishu_id,
            "admin_secret": "admin-secret-key-2025",
        }

        logger.info(f"正在通过飞书ID获取用户ID: feishu_id={feishu_id}")

        response = requests.post(url, json=data, timeout=10)

        if response.status_code == 200:
            result = cast(dict[str, Any], response.json())
            user_id = result.get("user_id")
            if user_id:
                logger.info(
                    f"成功获取用户ID: feishu_id={feishu_id} -> user_id={user_id}",
                )
                return user_id
            else:
                logger.warning(f"响应中没有找到用户ID: {result}")
                return None
        else:
            logger.warning(
                f"获取用户ID失败: status={response.status_code}, response={response.text}",
            )
            return None

    except Exception as e:
        logger.error(f"获取用户ID时出错: {e}")
        return None


# Sandbox helper functions
def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """在同步上下文中运行异步函数"""
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在运行的事件循环中，创建一个新任务
            import nest_asyncio  # type: ignore[import-untyped]

            nest_asyncio_mod = cast(Any, nest_asyncio)
            nest_asyncio_mod.apply()
            return loop.run_until_complete(coro)
        else:
            # 如果没有运行的事件循环，直接运行
            return loop.run_until_complete(coro)
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        return asyncio.run(coro)


def cleanup_temp_file(temp_file_path: str):
    """
    清理临时文件

    Args:
        temp_file_path: 临时文件路径
    """
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
            logger.info(f"已清理临时文件: {temp_file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")


def get_feishu_chat_list(
    sort_type: Annotated[
        str,
        "群组排序方式 - ByCreateTimeAsc/ByActiveTimeDesc (default ByCreateTimeAsc)",
    ] = "ByCreateTimeAsc",
    page_size: Annotated[
        int,
        "分页大小，限制一次请求返回的数据条目数 (default 20, max 100)",
    ] = 20,
    user_id_type: Annotated[
        str | None,
        "用户ID类型 - open_id/union_id/user_id (default open_id)",
    ] = None,
    page_token: Annotated[str | None, "分页标记，用于获取下一页数据"] = None,
) -> str:
    """获取用户或机器人所在的群列表。支持按创建时间或活跃时间排序，支持分页查询。"""
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        # 构造请求对象
        request_builder = ListChatRequest.builder().sort_type(sort_type).page_size(page_size)

        # 添加可选的用户ID类型
        if user_id_type:
            request_builder = request_builder.user_id_type(user_id_type)

        # 添加可选的分页标记
        if page_token:
            request_builder = request_builder.page_token(page_token)

        request: ListChatRequest = request_builder.build()

        # 发起请求
        response: ListChatResponse = client.im.v1.chat.list(request)

        # 处理失败返回
        if not response.success():
            error_msg = f"获取飞书聊天列表失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)

            # 尝试获取详细错误信息
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
            except Exception:
                pass

            return error_msg

        # 处理业务结果
        chat_data = _marshal_json(response.data, indent=4)
        return chat_data

    except BaseException as e:
        error_msg = f"获取飞书聊天列表失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def get_chat_user_mapping(chat_id: str) -> dict[str, str]:
    """获取指定群聊中用户ID和用户名的对应关系，支持分页获取所有用户

    Args:
        chat_id: 群聊ID

    Returns:
        dict: 用户ID到用户名的映射字典，格式为 {user_id: user_name}
    """
    try:
        user_mapping: dict[str, str] = {}
        page_token = None
        page_count = 0
        max_pages = 50  # 最大页数限制，避免无限循环

        while page_count < max_pages:
            page_count += 1

            # 获取当前页的群组成员列表
            logger.info(f"获取群聊 {chat_id} 第 {page_count} 页用户列表...")
            members_response = search_users_in_chat(
                chat_id=chat_id,
                page_size=100,
                page_token=page_token or None,
            )

            # 解析响应
            try:
                raw_members = json.loads(members_response)
            except json.JSONDecodeError:
                logger.warning(
                    f"解析群组成员数据失败 (第{page_count}页): {members_response}",
                )
                break
            try:
                members_data = _ensure_dict(raw_members, context=f"第{page_count}页群组成员数据")
            except ValueError as error:
                logger.warning(str(error))
                break

            # 处理当前页的用户数据
            if "users" in members_data:
                try:
                    current_page_users = _ensure_list_of_dicts(
                        members_data["users"],
                        context=f"第{page_count}页群组成员数据.users",
                    )
                except ValueError as error:
                    logger.warning(str(error))
                    current_page_users = []
                logger.info(
                    f"第 {page_count} 页获取到 {len(current_page_users)} 个用户",
                )

                for user in current_page_users:
                    user_id = user.get("member_id", "")
                    user_name = user.get("name", "未知用户")
                    if user_id:
                        user_mapping[user_id] = user_name

            # 检查是否还有下一页
            has_more = bool(members_data.get("has_more", False))
            page_token_raw = members_data.get("page_token", "")
            page_token = page_token_raw if isinstance(page_token_raw, str) else ""

            if not has_more or not page_token:
                logger.info(f"群聊 {chat_id} 用户列表获取完毕，共 {page_count} 页")
                break

        if page_count >= max_pages:
            logger.warning(
                f"群聊 {chat_id} 用户列表获取达到最大页数限制 {max_pages}，可能未获取完整",
            )

        logger.info(f"成功获取群聊 {chat_id} 的用户映射，共 {len(user_mapping)} 个用户")
        return user_mapping

    except Exception as e:
        logger.error(f"获取群聊用户映射失败: {e}")
        return {}


def truncate_message_content(content: str, max_length: int = 2000) -> str:
    """截断消息内容，如果超过指定长度则前后各取一部分，中间加省略号

    Args:
        content: 原始消息内容
        max_length: 最大长度，默认2000字符

    Returns:
        str: 截断后的消息内容
    """
    if len(content) <= max_length:
        return content

    # 计算前后各取的长度
    half_length = (max_length - 3) // 2  # 减去3是为了给省略号留空间

    return content[:half_length] + "..." + content[-half_length:]


def extract_post_text(content_obj: JSONDict) -> str:
    """从富文本消息对象中提取纯文本内容

    Args:
        content_obj: 富文本消息的内容对象

    Returns:
        str: 提取的纯文本内容
    """
    try:
        # text_parts = []

        # 检查是否是带语言标识的格式 (如 {"zh_cn": {...}})
        has_lang_format = any(
            key in content_obj
            for key in [
                "zh_cn",
                "en_us",
                "ja_jp",
            ]
        )

        if has_lang_format:
            # 处理带语言字段的格式
            for lang in ["zh_cn", "en_us", "ja_jp"]:
                if lang in content_obj:
                    lang_content = content_obj[lang]
                    extracted_text = _extract_post_content(lang_content)
                    if extracted_text and extracted_text != "[富文本消息]":
                        return extracted_text
        else:
            # 处理直接格式 (如 {"title": "...", "content": [...]})
            extracted_text = _extract_post_content(content_obj)
            if extracted_text and extracted_text != "[富文本消息]":
                return extracted_text

        return "[富文本消息]"

    except Exception as e:
        logger.warning(f"提取富文本内容失败: {e}")
        return "[富文本消息]"


def _extract_post_content(content_data: JSONDict) -> str:
    """从富文本内容数据中提取文本

    Args:
        content_data: 富文本内容数据

    Returns:
        str: 提取的文本内容
    """
    try:
        text_parts: list[str] = []

        # 提取标题
        title = content_data.get("title", "")
        if title:
            text_parts.append(f"标题: {title}")

        # 提取内容
        content_lines_raw = content_data.get("content", [])
        content_lines_raw_typed: list[Any] = []
        if not isinstance(content_lines_raw, list):
            return "[富文本消息]"
        content_lines_raw_typed = cast(list[Any], content_lines_raw)
        content_lines: list[list[JSONDict]] = []
        for line_obj in content_lines_raw_typed:
            if isinstance(line_obj, list):
                content_lines.append(cast(list[JSONDict], line_obj))

        for line in content_lines:
            line_text: str = ""
            for element in line:
                tag = str(element.get("tag", ""))
                if tag == "text":
                    text_value = element.get("text", "")
                    if isinstance(text_value, str):
                        line_text += text_value
                elif tag == "a":
                    text = element.get("text", "")
                    href = element.get("href", "")
                    text_str = text if isinstance(text, str) else ""
                    href_str = href if isinstance(href, str) else ""
                    if text_str and href_str:
                        line_text += f"{text_str}({href_str})"
                    elif text_str:
                        line_text += text_str
                    elif href_str:
                        line_text += href_str
                elif tag == "at":
                    user_name = element.get("user_name", "")
                    user_id = element.get("user_id", "")
                    user_name_str = user_name if isinstance(user_name, str) else ""
                    user_id_str = user_id if isinstance(user_id, str) else ""
                    if user_name_str:
                        line_text += f"@{user_name_str}"
                    elif user_id_str:
                        line_text += f"@{user_id_str}"
                    else:
                        line_text += "@某人"
                elif tag == "img":
                    line_text += "[图片]"
                elif tag == "media":
                    line_text += "[视频]"
                elif tag == "emotion":
                    emoji_type = element.get("emoji_type", "表情")
                    emoji_str = emoji_type if isinstance(emoji_type, str) else "表情"
                    line_text += f"[{emoji_str}]"

            if line_text.strip():
                text_parts.append(line_text.strip())

        return "\n".join(text_parts) if text_parts else "[富文本消息]"

    except Exception as e:
        logger.warning(f"提取富文本内容失败: {e}")
        return "[富文本消息]"


def extract_interactive_card_text(content_obj: JSONDict) -> str:
    """从交互卡片消息对象中提取纯文本内容

    Args:
        content_obj: 交互卡片消息的内容对象

    Returns:
        str: 提取的纯文本内容
    """
    try:
        text_parts: list[str] = []

        # 提取卡片标题
        # 方式1：标题在header对象中（完整的交互卡片格式）
        header: dict[str, Any] = {}
        header_raw = content_obj.get("header", {})
        if isinstance(header_raw, dict):
            header = cast(dict[str, Any], header_raw)
        if header:
            header_title: Any = header.get("title", {})
            if isinstance(header_title, dict):
                header_title_dict = cast(dict[str, Any], header_title)
                title_text = _first_str(
                    header_title_dict.get("content"),
                    header_title_dict.get("text"),
                )
                if title_text:
                    text_parts.append(f"卡片标题: {title_text}")
            elif isinstance(header_title, str) and header_title:
                text_parts.append(f"卡片标题: {header_title}")

        # 方式2：标题直接在根对象中（简化的交互卡片格式）
        if "title" in content_obj and not header:
            root_title: Any = content_obj.get("title", "")
            if isinstance(root_title, str) and root_title:
                text_parts.append(f"卡片标题: {root_title}")
            elif isinstance(root_title, dict):
                root_title_dict = cast(dict[str, Any], root_title)
                root_title_text = _first_str(
                    root_title_dict.get("content"),
                    root_title_dict.get("text"),
                )
                if root_title_text:
                    text_parts.append(f"卡片标题: {root_title_text}")

        # 提取元素内容
        elements_raw = content_obj.get("elements", [])
        elements_list: list[Any] = []
        if isinstance(elements_raw, list):
            elements_list = cast(list[Any], elements_raw)
        for element in elements_list:
            # 处理嵌套的元素结构，有些交互卡片的elements是嵌套数组
            if isinstance(element, list):
                # 如果元素是数组，递归处理每个子元素
                element_list: list[Any] = cast(list[Any], element)
                for sub_element in element_list:
                    if isinstance(sub_element, dict):
                        sub_element_typed: dict[str, Any] = cast(dict[str, Any], sub_element)
                        element_text = _extract_element_text(sub_element_typed)
                        if element_text:
                            text_parts.append(element_text)
            elif isinstance(element, dict):
                # 直接处理字典元素
                element_typed: dict[str, Any] = cast(dict[str, Any], element)
                element_text = _extract_element_text(element_typed)
                if element_text:
                    text_parts.append(element_text)

        return "\n".join(text_parts) if text_parts else "[交互卡片]"

    except Exception as e:
        logger.warning(f"提取交互卡片内容失败: {e}")
        return "[交互卡片]"


def _extract_element_text(element: JSONDict) -> str:
    """从卡片元素中提取文本内容

    Args:
        element: 卡片元素对象

    Returns:
        str: 提取的文本内容
    """
    try:
        tag = str(element.get("tag", ""))

        if tag == "text":
            # 纯文本元素（直接包含text字段）
            text = element.get("text", "")
            if isinstance(text, str) and text:
                return text

        elif tag == "div":
            # div 元素
            text_obj = element.get("text", {})
            if isinstance(text_obj, dict):
                text_obj_dict = cast(dict[str, Any], text_obj)
                div_content = _first_str(text_obj_dict.get("content"))
                if div_content:
                    return div_content
            elif isinstance(text_obj, str):
                return text_obj

        elif tag == "markdown" or tag == "lark_md":
            # markdown 元素
            markdown_content: Any = element.get("content", "")
            if isinstance(markdown_content, str) and markdown_content:
                return markdown_content

        elif tag == "plain_text":
            # 纯文本元素
            plain_text_content: Any = element.get("content", "")
            if isinstance(plain_text_content, str) and plain_text_content:
                return plain_text_content

        elif tag == "button":
            # 按钮元素
            text_obj = element.get("text", {})
            if isinstance(text_obj, dict):
                text_obj_dict = cast(dict[str, Any], text_obj)
                button_content = _first_str(text_obj_dict.get("content"))
                if button_content:
                    return f"[按钮: {button_content}]"
            elif isinstance(text_obj, str) and text_obj:
                return f"[按钮: {text_obj}]"

        elif tag == "column_set":
            # 列集合
            columns_raw = element.get("columns", [])
            columns: list[dict[str, Any]] = []
            columns_raw_typed: list[Any] = []
            if isinstance(columns_raw, list):
                columns_raw_typed = cast(list[Any], columns_raw)
            for column_obj in columns_raw_typed:
                if isinstance(column_obj, dict):
                    columns.append(cast(dict[str, Any], column_obj))
            column_texts: list[str] = []
            for column in columns:
                column_elements_raw = column.get("elements", [])
                column_elements: list[dict[str, Any]] = []
                column_elements_list: list[Any] = []
                if isinstance(column_elements_raw, list):
                    column_elements_list = cast(list[Any], column_elements_raw)
                for elem in column_elements_list:
                    if isinstance(elem, dict):
                        column_elements.append(cast(dict[str, Any], elem))
                for col_element in column_elements:
                    col_text = _extract_element_text(col_element)
                    if col_text:
                        column_texts.append(col_text)
            if column_texts:
                return " | ".join(column_texts)

        elif tag == "field":
            # 字段元素
            name = element.get("name", "")
            text_obj = element.get("text", {})
            field_content: str = ""
            if isinstance(text_obj, dict):
                text_obj_dict = cast(dict[str, Any], text_obj)
                field_content = _first_str(text_obj_dict.get("content"))
            elif isinstance(text_obj, str):
                field_content = text_obj

            name_str = name if isinstance(name, str) else ""
            if name_str and field_content:
                return f"{name_str}: {field_content}"
            if field_content:
                return field_content

        elif tag == "img":
            # 图片元素
            alt = element.get("alt", {})
            if isinstance(alt, dict):
                alt_dict = cast(dict[str, Any], alt)
                alt_text = _first_str(alt_dict.get("content"))
                if alt_text:
                    return f"[图片: {alt_text}]"
            return "[图片]"

        elif tag == "action":
            # 动作元素
            actions_raw = element.get("actions", [])
            actions: list[dict[str, Any]] = []
            actions_raw_list: list[Any] = []
            if isinstance(actions_raw, list):
                actions_raw_list = cast(list[Any], actions_raw)
            for action in actions_raw_list:
                if isinstance(action, dict):
                    actions.append(cast(dict[str, Any], action))
            action_texts: list[str] = []
            for action in actions:
                action_text = _extract_element_text(action)
                if action_text:
                    action_texts.append(action_text)
            if action_texts:
                return " ".join(action_texts)

        # 递归处理嵌套元素
        if "elements" in element:
            nested_elements_raw = element.get("elements", [])
            nested_elements: list[dict[str, Any]] = []
            nested_elements_list: list[Any] = []
            if isinstance(nested_elements_raw, list):
                nested_elements_list = cast(list[Any], nested_elements_raw)
            for nested_element in nested_elements_list:
                if isinstance(nested_element, dict):
                    nested_elements.append(cast(dict[str, Any], nested_element))
            nested_texts: list[str] = []
            for nested_element in nested_elements:
                nested_text = _extract_element_text(nested_element)
                if nested_text:
                    nested_texts.append(nested_text)
            if nested_texts:
                return "\n".join(nested_texts)

        return ""

    except Exception as e:
        logger.warning(f"提取元素文本失败: {e}")
        return ""


def get_feishu_message_list(
    container_id: Annotated[str, "Container ID - 群聊或单聊的ID，或话题ID"],
    container_id_type: Annotated[
        str,
        "Container type: 'chat' for 单聊/群聊, 'thread' for 话题 (default 'chat')",
    ] = "chat",
    page_size: Annotated[
        int,
        "分页大小，单次请求返回的数据条目数 (default 20, range 1-50)",
    ] = 20,
    sort_type: Annotated[
        str,
        "排序方式: ByCreateTimeAsc 或 ByCreateTimeDesc (default ByCreateTimeAsc)",
    ] = "ByCreateTimeAsc",
    start_time: Annotated[
        str | None,
        "起始时间，支持格式：'2025-06-12 20:17:00' 或 '2025-06-12'，获取指定时间范围内的消息",
    ] = None,
    end_time: Annotated[
        str | None,
        "结束时间，支持格式：'2025-06-12 20:17:00' 或 '2025-06-12'，获取指定时间范围内的消息",
    ] = None,
    page_token: Annotated[str | None, "分页标记，用于获取下一页数据"] = None,
    format_messages: Annotated[
        bool,
        "是否格式化消息显示用户名和时间 (default True)",
    ] = True,
) -> str:
    """获取指定会话(单聊/群聊/话题)内的历史消息。注意：机器人必须在被查询的群组中才能获取消息。"""
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        # 时间格式转换函数
        def convert_to_timestamp(time_str: str) -> str:
            """将datetime格式字符串转换为秒级时间戳"""
            try:
                # 尝试解析 '2025-06-12 20:17:00' 格式
                if len(time_str) > 10:
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                else:
                    # 尝试解析 '2025-06-12' 格式，默认为当天00:00:00
                    dt = datetime.strptime(time_str, "%Y-%m-%d")

                # 转换为秒级时间戳
                timestamp = int(dt.timestamp())
                return str(timestamp)
            except ValueError:
                # 如果转换失败，假设已经是时间戳格式，直接返回
                return time_str

        # 处理时间参数
        final_start_time = None
        final_end_time = None

        if start_time:
            final_start_time = convert_to_timestamp(start_time)
        if end_time:
            final_end_time = convert_to_timestamp(end_time)

        # 构造请求对象
        request_builder = (
            ListMessageRequest.builder()
            .container_id_type(container_id_type)
            .container_id(container_id)
            .sort_type(sort_type)
            .page_size(page_size)
        )

        # 添加可选参数
        if final_start_time:
            request_builder = request_builder.start_time(final_start_time)
        if final_end_time:
            request_builder = request_builder.end_time(final_end_time)
        if page_token:
            request_builder = request_builder.page_token(page_token)

        request: ListMessageRequest = request_builder.build()

        # 发起请求
        response: ListMessageResponse = client.im.v1.message.list(request)

        # 处理失败返回
        if not response.success():
            error_msg = f"获取飞书消息列表失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)

            # 尝试获取详细错误信息
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
            except Exception:
                pass

            return error_msg

        # 处理业务结果
        raw_message_data = _marshal_json(response.data, indent=4)

        # 如果不需要格式化，直接返回原始数据
        if not format_messages:
            return raw_message_data

        # 解析消息数据进行格式化
        try:
            raw_message_json = json.loads(raw_message_data)
        except json.JSONDecodeError:
            logger.warning("消息数据解析失败，返回原始数据")
            return raw_message_data
        try:
            message_json = _ensure_dict(raw_message_json, context="消息数据")
        except ValueError:
            logger.warning("消息数据应为字典格式，返回原始数据")
            return raw_message_data

        # 获取用户映射（仅对群聊）
        user_mapping: dict[str, str] = {}
        if container_id_type == "chat":
            user_mapping = get_chat_user_mapping(container_id)

        # 格式化消息列表
        formatted_messages: list[str] = []

        if "items" in message_json:
            try:
                message_items = _ensure_list_of_dicts(message_json["items"], context="消息列表items")
            except ValueError as error:
                logger.warning(str(error))
                message_items = []
            for message in message_items:
                try:
                    # 获取消息基本信息
                    create_time = message.get("create_time", "")
                    sender_raw = message.get("sender", {})
                    sender: dict[str, Any] = (
                        _ensure_dict(cast(dict[str, Any], sender_raw), context="消息发送者") if isinstance(sender_raw, dict) else {}
                    )
                    sender_id = sender.get("id", "")
                    sender_type = sender.get("sender_type", "")
                    msg_type = message.get("msg_type", "")
                    content_text: str = ""

                    # 解析消息内容
                    body_raw = message.get("body", {})
                    body: dict[str, Any] = (
                        _ensure_dict(cast(dict[str, Any], body_raw), context="消息体") if isinstance(body_raw, dict) else {}
                    )
                    body_content = body.get("content", "")
                    content = body_content if isinstance(body_content, str) else json.dumps(body_content, ensure_ascii=False)
                    if content:
                        try:
                            content_obj_raw = json.loads(content)
                            if not isinstance(content_obj_raw, dict):
                                content_text = str(content_obj_raw)
                                content_obj: dict[str, Any] = {}
                            else:
                                content_obj = cast(dict[str, Any], content_obj_raw)

                                # 根据消息类型提取文本内容
                                if msg_type == "text":
                                    content_text = str(content_obj.get("text", ""))
                                elif msg_type == "post":
                                    # 富文本消息，尝试提取纯文本
                                    content_text = extract_post_text(content_obj)
                                elif msg_type == "image":
                                    content_text = "[图片]"
                                elif msg_type == "file":
                                    content_text = "[文件]"
                                elif msg_type == "audio":
                                    content_text = "[语音]"
                                elif msg_type == "media":
                                    content_text = "[视频]"
                                elif msg_type == "sticker":
                                    content_text = "[表情包]"
                                elif msg_type == "interactive":
                                    # 交互卡片消息，尝试提取文本内容
                                    content_text = extract_interactive_card_text(
                                        content_obj,
                                    )
                                elif msg_type == "system":
                                    # 系统消息，尝试提取实际内容
                                    if content_obj.get("text"):
                                        content_text = str(
                                            content_obj.get(
                                                "text",
                                                "",
                                            ),
                                        )
                                    elif content_obj.get("content"):
                                        content_text = str(
                                            content_obj.get(
                                                "content",
                                                "",
                                            ),
                                        )
                                    elif content_obj.get("message"):
                                        content_text = str(
                                            content_obj.get(
                                                "message",
                                                "",
                                            ),
                                        )
                                    elif content_obj.get("template"):
                                        # 飞书系统消息通常有template字段，尝试格式化
                                        template_value = content_obj.get("template", "")
                                        template = template_value if isinstance(template_value, str) else str(template_value)
                                        from_user_raw = content_obj.get("from_user", [])
                                        to_chatters_raw = content_obj.get("to_chatters", [])
                                        from_user: list[str] = []
                                        to_chatters: list[str] = []
                                        if isinstance(from_user_raw, list):
                                            from_user_raw_list: list[Any] = cast(list[Any], from_user_raw)
                                            for user in from_user_raw_list:
                                                from_user.append(str(user))
                                        if isinstance(to_chatters_raw, list):
                                            to_chatters_raw_list: list[Any] = cast(list[Any], to_chatters_raw)
                                            for chatter in to_chatters_raw_list:
                                                to_chatters.append(str(chatter))

                                        # 替换常见的模板变量
                                        if from_user:
                                            template = template.replace(
                                                "{from_user}",
                                                ", ".join(
                                                    from_user,
                                                ),
                                            )
                                        if to_chatters:
                                            template = template.replace(
                                                "{to_chatters}",
                                                ", ".join(
                                                    to_chatters,
                                                ),
                                            )

                                        # 处理其他可能的模板变量
                                        for key, value in content_obj.items():
                                            if key not in [
                                                "template",
                                                "from_user",
                                                "to_chatters",
                                                "divider_text",
                                            ] and isinstance(value, str):
                                                template = template.replace(
                                                    f"{{{key}}}",
                                                    value,
                                                )

                                        # 移除未替换的模板变量
                                        import re

                                        template = re.sub(
                                            r"\{[^}]*\}",
                                            "[信息不可用]",
                                            template,
                                        )

                                        content_text = template
                                    else:
                                        content_text = str(content_obj)
                                else:
                                    content_text = f"[{msg_type}类型消息]"

                        except json.JSONDecodeError:
                            content_text = content  # 如果解析失败，直接使用原始内容

                    # 格式化时间
                    formatted_time = ""
                    if create_time:
                        try:
                            # 将毫秒时间戳转换为可读时间
                            timestamp = int(create_time) / 1000  # 飞书API返回的是毫秒时间戳
                            dt = datetime.fromtimestamp(timestamp)
                            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            formatted_time = create_time

                    # 获取用户名
                    user_name = "未知用户"
                    if sender_type == "user" and sender_id:
                        user_name = user_mapping.get(
                            sender_id,
                            f"用户ID:{sender_id}",
                        )
                        # 如果用户映射中找不到，可能是权限问题或新用户
                        if user_name == f"用户ID:{sender_id}":
                            logger.debug(
                                f"User mapping not found for sender_id: {sender_id}",
                            )
                    elif sender_type == "app":
                        user_name = "机器人"
                    elif msg_type == "system":
                        user_name = "系统"
                    elif not sender_type and not sender_id:
                        # 没有发送者信息，可能是系统消息
                        user_name = "系统" if msg_type == "system" else "未知用户"
                    else:
                        # 其他情况，显示更多调试信息
                        user_name = f"未知用户(类型:{sender_type})"

                    # 截断长消息内容
                    if content_text:
                        content_text = truncate_message_content(content_text)

                    # 格式化单条消息
                    formatted_message = f"{formatted_time} 【{user_name}】:\n{content_text}"
                    formatted_messages.append(formatted_message)

                except Exception as e:
                    logger.warning(f"格式化单条消息失败: {e}")
                    # 出错时添加原始消息标识
                    formatted_messages.append(f"[消息解析错误: {str(e)}]")

        # 构建最终结果
        result: dict[str, Any] = {"formatted_messages": formatted_messages}

        return json.dumps(result, indent=4, ensure_ascii=False)

    except BaseException as e:
        error_msg = f"获取飞书消息列表失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def send_feishu_message(
    receive_id: Annotated[str, "接收者的ID - 用户ID或群组ID"],
    content: Annotated[str | dict[str, Any] | list[Any], "消息内容 - 详见下方使用说明"],
    msg_type: Annotated[
        str,
        "消息类型 - text/post/image/interactive/audio/media/file/share_chat/share_user/sticker",
    ] = "text",
    receive_id_type: Annotated[
        str,
        "接收者ID类型 - open_id/user_id/union_id/email/chat_id",
    ] = "open_id",
) -> str:
    """发送消息到指定用户或群组。统一的消息发送接口，支持多种消息类型。

    参数说明：
    - receive_id: 接收者ID（群组ID一般以oc_开头，用户ID以ou_开头）
    - content: 消息内容，根据msg_type有不同格式要求
    - msg_type: 消息类型，支持text/post/image/interactive/audio/media/file/share_chat/share_user/sticker
    - receive_id_type: ID类型，通常用chat_id发送到群组，用open_id发送给用户

    使用示例：

    1. 发送文本消息：
    msg_type="text"
    content="你好，这是一条文本消息"

    2. 发送文本消息（带@功能）：
    msg_type="text"
    content='{"text":"<at user_id=\\"ou_xxxxxxxxx\\">张三</at> 你好！"}'
    # 或者直接传入文本内容（推荐）：
    content="<at user_id=\"ou_xxxxxxxxx\">张三</at> 你好！"

    3. 发送富文本消息：
    msg_type="post"
    content='{"zh_cn":{"title":"通知标题","content":[[{"tag":"text","text":"第一行："},{"tag":"a","href":"https://example.com","text":"链接文字"}],[{"tag":"text","text":"第二行：普通文本"}]]}}'

    4. 发送图片消息：
    msg_type="image"
    content='{"image_key":"img_xxxxxxxxx"}'  # 需要先调用upload_feishu_image获取image_key

    5. 发送语音消息：
    msg_type="audio"
    content='{"file_key":"file_v2_xxxxxxxxx"}'  # 需要先调用upload_feishu_file获取file_key

    6. 发送视频消息：
    msg_type="media"
    content='{"file_key":"file_v2_xxxxxxxxx","image_key":"img_xxxxxxxxx"}'  # 视频文件和封面图片

    7. 发送文件消息：
    msg_type="file"
    content='{"file_key":"file_v2_xxxxxxxxx"}'

    8. 发送群名片：
    msg_type="share_chat"
    content='{"chat_id":"oc_xxxxxxxxx"}'

    9. 发送用户名片：
    msg_type="share_user"
    content='{"user_id":"ou_xxxxxxxxx"}'  # 只支持open_id

    10. 发送表情包：
    msg_type="sticker"
    content='{"file_key":"file_v2_xxxxxxxxx"}'  # 目前仅支持发送机器人收到的表情包

    11. 发送交互卡片：
    msg_type="interactive"
    content='{"config":{"wide_screen_mode":true},"elements":[{"tag":"div","text":{"content":"卡片内容","tag":"lark_md"}}]}'

    富文本格式详解：
    - 文本元素：{"tag":"text","text":"文字内容"}
    - 链接元素：{"tag":"a","href":"链接地址","text":"显示文字"}
    - @用户：{"tag":"at","user_id":"用户ID","user_name":"用户名"}
    - 图片：{"tag":"img","image_key":"图片key"}
    - 视频：{"tag":"media","file_key":"视频key","image_key":"封面key"}
    - 表情：{"tag":"emotion","emoji_type":"SMILE"}
    - 换行：每个数组元素代表一行，行内元素并排显示

    注意事项：
    - 机器人必须在被发送消息的群组中（针对群组消息）
    - 机器人必须对用户可见（针对私聊消息）
    - 所有文件、图片需要先通过对应的上传接口获取key
    - 表情包目前仅支持转发机器人收到的表情包
    """
    try:
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        final_content: str = ""

        # 处理不同类型的消息内容
        if msg_type == "text":
            # 文本消息处理
            if isinstance(content, str):
                try:
                    # 验证JSON格式
                    json.loads(content)
                    # 如果是JSON格式，直接使用（可能包含富文本格式如@用户）
                    final_content = content
                except json.JSONDecodeError:
                    # 如果不是JSON格式，包装成文本消息格式
                    final_content = json.dumps({"text": content}, ensure_ascii=False)
            else:
                final_content = json.dumps({"text": content}, ensure_ascii=False)

        elif msg_type == "post":
            # 富文本消息处理
            try:
                if isinstance(content, str):
                    # 验证JSON格式
                    content_obj = json.loads(content)
                    final_content = content
                else:
                    # 字典转JSON字符串
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证富文本消息的基本结构
                content_obj = json.loads(final_content)
                if not isinstance(content_obj, dict):
                    raise ValueError("富文本消息必须是字典格式")

                # 检查是否包含必要的语言字段
                has_lang = any(key in content_obj for key in ["zh_cn", "en_us", "ja_jp"])
                if not has_lang:
                    raise ValueError(
                        "富文本消息必须包含至少一种语言版本（zh_cn/en_us/ja_jp）",
                    )

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = (
                    f'富文本消息格式错误: {str(e)}\n期望格式: {{"zh_cn":{{"title":"标题","content":[[{{"tag":"text","text":"内容"}}]]}}}}'
                )
                logger.error(error_msg)
                return error_msg

        elif msg_type == "image":
            # 图片消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证图片消息格式
                content_obj_raw = json.loads(final_content)
                if not isinstance(content_obj_raw, dict) or "image_key" not in content_obj_raw:
                    raise ValueError("图片消息必须包含image_key字段")
                content_obj = cast(dict[str, Any], content_obj_raw)

                # 验证image_key格式
                image_key = content_obj.get("image_key")
                if not isinstance(image_key, str) or not image_key:
                    raise ValueError("image_key必须是非空字符串")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f'图片消息格式错误: {str(e)}\n期望格式: {{"image_key":"img_xxxxxxxxx"}}'
                logger.error(error_msg)
                return error_msg

        elif msg_type == "audio":
            # 语音消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证语音消息格式
                content_obj = json.loads(final_content)
                if not isinstance(content_obj, dict) or "file_key" not in content_obj:
                    raise ValueError("语音消息必须包含file_key字段")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f'语音消息格式错误: {str(e)}\n期望格式: {{"file_key":"file_v2_xxxxxxxxx"}}'
                logger.error(error_msg)
                return error_msg

        elif msg_type == "media":
            # 视频消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证视频消息格式
                content_obj = json.loads(final_content)
                if not isinstance(content_obj, dict) or "file_key" not in content_obj:
                    raise ValueError("视频消息必须包含file_key字段")

                # image_key是可选的，用于视频封面
                if "image_key" in content_obj and not content_obj["image_key"]:
                    logger.warning("视频消息的image_key为空，将不显示封面图片")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f'视频消息格式错误: {str(e)}\n期望格式: {{"file_key":"file_v2_xxxxxxxxx","image_key":"img_xxxxxxxxx"}}'
                logger.error(error_msg)
                return error_msg

        elif msg_type == "file":
            # 文件消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证文件消息格式
                content_obj = json.loads(final_content)
                if not isinstance(content_obj, dict) or "file_key" not in content_obj:
                    raise ValueError("文件消息必须包含file_key字段")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f'文件消息格式错误: {str(e)}\n期望格式: {{"file_key":"file_v2_xxxxxxxxx"}}'
                logger.error(error_msg)
                return error_msg

        elif msg_type == "share_chat":
            # 群名片消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证群名片消息格式
                content_obj_raw = json.loads(final_content)
                if not isinstance(content_obj_raw, dict) or "chat_id" not in content_obj_raw:
                    raise ValueError("群名片消息必须包含chat_id字段")
                content_obj = cast(dict[str, Any], content_obj_raw)

                # 验证chat_id格式
                chat_id = content_obj.get("chat_id")
                if not isinstance(chat_id, str) or not chat_id:
                    raise ValueError("chat_id必须是非空字符串")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f'群名片消息格式错误: {str(e)}\n期望格式: {{"chat_id":"oc_xxxxxxxxx"}}'
                logger.error(error_msg)
                return error_msg

        elif msg_type == "share_user":
            # 用户名片消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证用户名片消息格式
                content_obj_raw = json.loads(final_content)
                if not isinstance(content_obj_raw, dict) or "user_id" not in content_obj_raw:
                    raise ValueError("用户名片消息必须包含user_id字段")
                content_obj = cast(dict[str, Any], content_obj_raw)

                # 验证user_id格式（只支持open_id）
                user_id = content_obj.get("user_id")
                if not isinstance(user_id, str) or not user_id:
                    raise ValueError("user_id必须是非空字符串")
                if not user_id.startswith("ou_"):
                    logger.warning(
                        "用户名片消息的user_id建议使用open_id格式（以ou_开头）",
                    )

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f'用户名片消息格式错误: {str(e)}\n期望格式: {{"user_id":"ou_xxxxxxxxx"}}'
                logger.error(error_msg)
                return error_msg

        elif msg_type == "sticker":
            # 表情包消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证表情包消息格式
                content_obj = json.loads(final_content)
                if not isinstance(content_obj, dict) or "file_key" not in content_obj:
                    raise ValueError("表情包消息必须包含file_key字段")

                # 提示表情包的限制
                logger.info("注意：表情包消息目前仅支持发送机器人收到的表情包")

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = (
                    f'表情包消息格式错误: {str(e)}\n期望格式: {{"file_key":"file_v2_xxxxxxxxx"}}\n注意：目前仅支持发送机器人收到的表情包'
                )
                logger.error(error_msg)
                return error_msg

        elif msg_type == "interactive":
            # 交互卡片消息处理
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)

                # 验证交互卡片的基本结构
                content_obj = json.loads(final_content)
                if not isinstance(content_obj, dict):
                    raise ValueError("交互卡片必须是字典格式")

                # 检查是否包含必要的elements字段
                if "elements" not in content_obj:
                    raise ValueError("交互卡片必须包含elements字段")

                if not isinstance(content_obj["elements"], list):
                    raise ValueError("elements字段必须是数组格式")

            except (json.JSONDecodeError, ValueError) as e:
                expected_format = """
                {"config":{"wide_screen_mode":true},"elements":[{"tag":"div","text":{"content":"内容","tag":"lark_md"}}]}'
                """
                error_msg = f"交互卡片消息格式错误: {str(e)}\n期望格式:\n{expected_format}"
                logger.error(error_msg)
                return error_msg
        else:
            # 其他消息类型，尝试作为通用JSON处理
            try:
                if isinstance(content, str):
                    # 验证是否为有效JSON
                    json.loads(content)
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)
            except json.JSONDecodeError:
                # 如果不是有效JSON，作为普通字符串处理
                logger.warning(f"未知消息类型 {msg_type}，将content作为原始字符串处理")
                final_content = str(content)

        # 调试日志：记录最终发送的内容（截断显示）
        logger.info(
            f"发送消息类型: {msg_type}, 目标: {receive_id_type}={receive_id}, 内容长度: {len(final_content)}",
        )
        logger.debug(f"消息内容预览: {final_content[:200]}...")

        # 构造请求对象
        request_builder = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder().receive_id(receive_id).msg_type(msg_type).content(final_content).build(),
            )
        )

        request = request_builder.build()

        # 发起请求
        response: CreateMessageResponse = client.im.v1.message.create(request)

        # 处理失败返回
        if not response.success():
            error_msg = f"发送飞书消息失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)

            # 尝试获取详细错误信息
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"

                # 根据错误码提供更友好的提示
                error_code = response.code
                if error_code == 230002:
                    error_msg += "\n提示: 机器人不在目标群组中，请先将机器人添加到群组"
                elif error_code == 230004:
                    error_msg += "\n提示: 用户不存在或机器人对用户不可见"
                elif error_code == 1248010:
                    error_msg += "\n提示: 消息内容格式错误，请检查content字段格式"
                elif error_code == 9499:
                    error_msg += "\n提示: 应用权限不足，请检查机器人权限配置"

            except Exception:
                pass

            return error_msg

        # 处理业务结果
        result_data = _marshal_json(response.data, indent=4)

        # 解析返回的消息信息
        try:
            result_obj = cast(dict[str, Any], json.loads(result_data))
            message_id = result_obj.get("message_id", "")
            # create_time = result_obj.get('create_time', '')

            success_msg = "✅ 消息发送成功!"
            success_msg += f"\n📝 消息类型: {msg_type}"
            success_msg += f"\n🆔 消息ID: {message_id}"

            logger.info(f"消息发送成功: {message_id}")
            return success_msg

        except Exception:
            return f"✅ 消息发送成功!\n{result_data}"

    except BaseException as e:
        error_msg = f"❌ 发送飞书消息失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def upload_feishu_image(
    image_path: Annotated[str, "图片文件路径"],
    image_type: Annotated[str, "图片类型 - message (default message)"] = "message",
) -> str:
    """上传图片到飞书，返回image_key用于发送图片消息。

    自动检测文件位置：
    1. 首先尝试从沙盒下载文件
    2. 如果沙盒中不存在，则使用本地文件
    3. 自动管理沙盒生命周期和临时文件清理
    """
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        # 初始化变量
        actual_file_path = image_path

        try:
            # 读取图片文件
            with open(actual_file_path, "rb") as file_content:
                # 构造请求对象
                request: CreateImageRequest = (
                    CreateImageRequest.builder()
                    .request_body(
                        CreateImageRequestBody.builder().image_type(image_type).image(file_content).build(),
                    )
                    .build()
                )

                # 发起请求
                response: CreateImageResponse = client.im.v1.image.create(
                    request,
                )

                # 处理失败返回
                if not response.success():
                    error_msg = f"上传飞书图片失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
                    logger.error(error_msg)
                    return error_msg

                # 处理业务结果
                result_data = _marshal_json(response.data, indent=4)
                return f"图片上传成功!\n{result_data}"
        except Exception as e:
            error_msg = f"上传飞书图片失败. Error: {repr(e)}"
            logger.error(error_msg)
            return error_msg

    except BaseException as e:
        error_msg = f"上传飞书图片失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg

    # 注意：不再销毁沙盒，因为沙盒是从state中获取的，应该由外部管理


def upload_feishu_file(
    file_path: Annotated[str, "文件路径"],
    file_type: Annotated[str, "文件类型 - opus/mp4/pdf/doc/xls/ppt/stream等"],
    # file_name: Annotated[
    #     Optional[str], "文件名，如不提供则使用文件路径中的文件名"
    # ] = None,
    duration: Annotated[int | None, "音视频文件时长（秒），仅音视频文件需要"] = None,
) -> str:
    """上传文件到飞书，返回file_key用于发送文件消息。

    参数说明：
    - file_path: 文件路径
    - file_type: 文件类型，支持：
        * opus: 音频文件
        * mp4: 视频文件
        * pdf: PDF文档
        * doc: Word文档
        * xls: Excel文档
        * ppt: PowerPoint文档
        * stream: 其他类型文件
    - duration: 音视频文件时长（秒），音视频文件必须提供

    返回格式：
    成功时返回包含file_key的JSON字符串，可用于send_feishu_message发送文件消息
    """
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        # 初始化变量
        actual_file_path = file_path

        # 获取文件名
        # if not file_name:
        #     file_name = os.path.basename(file_path)

        # 音视频文件时长验证
        if file_type in ["opus", "mp4"] and duration is None:
            return json.dumps(
                {
                    "error": "参数错误",
                    "message": f"音视频文件类型 '{file_type}' 必须提供 duration 参数",
                },
                ensure_ascii=False,
            )

        try:
            # 读取文件内容
            with open(actual_file_path, "rb") as file_content:
                # 构造请求对象
                request_body_builder = CreateFileRequestBody.builder().file_type(file_type).file_name(file_path).file(file_content)

                # 添加可选的时长参数
                if duration is not None:
                    request_body_builder = request_body_builder.duration(
                        duration,
                    )

                request: CreateFileRequest = CreateFileRequest.builder().request_body(request_body_builder.build()).build()

                # 发起请求
                response: CreateFileResponse = client.im.v1.file.create(
                    request,
                )

                # 处理失败返回
                if not response.success():
                    error_msg = f"上传飞书文件失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
                    logger.error(error_msg)

                    # 尝试获取详细错误信息
                    try:
                        error_detail = json.loads(response.raw.content)
                        error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
                    except Exception:
                        pass

                    return json.dumps(
                        {"error": "上传失败", "message": error_msg},
                        ensure_ascii=False,
                    )

                # 处理业务结果
                result_data = _marshal_json(response.data, indent=4)
                return f"文件上传成功!\n{result_data}"
        except Exception as e:
            error_msg = f"上传飞书文件失败. Error: {repr(e)}"
            logger.error(error_msg)
            return error_msg

    except BaseException as e:
        error_msg = f"上传飞书文件失败. Error: {repr(e)}"
        logger.error(error_msg)
        return json.dumps(
            {"error": "上传异常", "message": error_msg},
            ensure_ascii=False,
        )


def reply_to_feishu_message(
    message_id: Annotated[str, "要回复的消息ID"],
    content: Annotated[str | dict[str, Any] | list[Any], "回复消息内容"],
    msg_type: Annotated[
        str,
        "回复消息类型 - text/post/image/interactive/audio/media/file/share_chat/share_user/sticker",
    ] = "text",
) -> str:
    """回复指定的消息。支持回复多种类型的消息内容。

    参数说明：
    - message_id: 要回复的消息ID
    - content: 回复消息内容，格式与send_feishu_message相同
    - msg_type: 回复消息类型，支持所有send_feishu_message支持的类型

    使用示例请参考send_feishu_message函数的文档说明，content格式完全一致。
    """
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        final_content: str = ""

        # 处理不同类型的消息内容（与send_feishu_message保持一致）
        if msg_type == "text":
            if isinstance(content, str):
                try:
                    # 验证JSON格式
                    json.loads(content)
                    # 如果是JSON格式，直接使用（可能包含富文本格式如@用户）
                    final_content = content
                except json.JSONDecodeError:
                    # 如果不是JSON格式，包装成文本消息格式
                    final_content = json.dumps({"text": content}, ensure_ascii=False)
            else:
                final_content = json.dumps({"text": content}, ensure_ascii=False)
        elif msg_type in [
            "post",
            "image",
            "audio",
            "media",
            "file",
            "share_chat",
            "share_user",
            "sticker",
            "interactive",
        ]:
            # 对于其他消息类型，确保是有效的JSON格式
            try:
                if isinstance(content, str):
                    json.loads(content)  # 验证JSON格式
                    final_content = content
                else:
                    final_content = json.dumps(content, ensure_ascii=False)
            except json.JSONDecodeError as e:
                error_msg = f"回复消息格式错误: {str(e)}\n消息类型: {msg_type}"
                logger.error(error_msg)
                return error_msg
        else:
            # 未知消息类型
            logger.warning(f"未知回复消息类型 {msg_type}，将content作为原始内容处理")
            final_content = str(content)

        logger.info(
            f"回复消息类型: {msg_type}, 目标消息ID: {message_id}, 内容长度: {len(final_content)}",
        )

        # 构造请求对象 - 根据飞书官方文档的正确格式
        request: ReplyMessageRequest = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder().msg_type(msg_type).content(final_content).build(),
            )
            .build()
        )

        # 发起请求
        response: ReplyMessageResponse = client.im.v1.message.reply(request)

        # 处理失败返回
        if not response.success():
            error_msg = f"回复飞书消息失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)

            # 尝试获取详细错误信息
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"

                # 根据错误码提供更友好的提示
                error_code = response.code
                if error_code == 230002:
                    error_msg += "\n提示: 机器人不在目标群组中"
                elif error_code == 230004:
                    error_msg += "\n提示: 目标消息不存在或机器人无权限访问"
                elif error_code == 1248010:
                    error_msg += "\n提示: 回复消息内容格式错误"

            except Exception:
                pass

            return error_msg

        # 处理业务结果
        result_data = _marshal_json(response.data, indent=4)

        # 解析返回的消息信息
        try:
            result_obj = cast(dict[str, Any], json.loads(result_data))
            reply_message_id = result_obj.get("message_id", "")
            create_time = result_obj.get("create_time", "")

            success_msg = "✅ 消息回复成功!"
            success_msg += f"\n📝 回复类型: {msg_type}"
            success_msg += f"\n🎯 原消息ID: {message_id}"
            success_msg += f"\n🆔 回复消息ID: {reply_message_id}"
            success_msg += f"\n⏰ 回复时间: {create_time}"

            logger.info(f"消息回复成功: {reply_message_id}")
            return success_msg

        except Exception:
            return f"✅ 消息回复成功!\n{result_data}"

    except BaseException as e:
        error_msg = f"❌ 回复飞书消息失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def search_users_in_chat(
    chat_id: Annotated[str, "群组ID，获取该群组中所有用户信息"],
    member_id_type: Annotated[
        str,
        "成员ID类型 - user_id/union_id/open_id/app_id (default open_id)",
    ] = "open_id",
    page_size: Annotated[
        int,
        "分页大小，单次请求返回的数据条目数 (default 100, max 100)",
    ] = 100,
    page_token: Annotated[str | None, "分页标记，用于获取下一页数据"] = None,
) -> str:
    """获取指定群组中所有用户的信息列表。"""
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client，启用token配置
        client = _create_lark_client(final_app_id, final_app_secret)

        # 构造获取群组成员的请求
        request_builder = GetChatMembersRequest.builder().chat_id(chat_id).member_id_type(member_id_type).page_size(page_size)

        # 添加可选的分页标记
        if page_token:
            request_builder = request_builder.page_token(page_token)

        request: GetChatMembersRequest = request_builder.build()

        # 发起请求获取群组成员
        response: GetChatMembersResponse = client.im.v1.chat_members.get(
            request,
        )

        # 处理失败返回
        if not response.success():
            error_msg = f"获取群组成员失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)

            # 尝试获取详细错误信息
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
            except Exception:
                pass

            return error_msg

        # 解析成员数据
        members_data = cast(dict[str, Any], json.loads(_marshal_json(response.data, indent=4)))
        user_list: list[dict[str, str]] = []

        if "items" in members_data:
            # 处理所有成员信息，返回简洁的用户列表
            members_items = cast(list[dict[str, Any]], members_data["items"])
            for member in members_items:
                user_info = {
                    "member_id": member.get("member_id", ""),
                    "member_id_type": member.get("member_id_type", "open_id"),
                    "name": member.get("name", ""),
                }
                user_list.append(user_info)

        # 构造返回结果 - 简洁的列表格式
        result: dict[str, Any] = {
            "member_total": len(user_list),
            "users": user_list,
            "has_more": members_data.get("has_more", False),
            "page_token": members_data.get("page_token", ""),
        }

        return json.dumps(result, indent=4, ensure_ascii=False)

    except BaseException as e:
        error_msg = f"获取群组用户信息失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def get_user_info_by_id(
    user_id: Annotated[str, "用户ID"],
    user_id_type: Annotated[
        str,
        "用户ID类型 - open_id/user_id/union_id (default open_id)",
    ] = "open_id",
    department_id_type: Annotated[
        str,
        "部门ID类型 - department_id/open_department_id (default open_department_id)",
    ] = "open_department_id",
) -> str:
    """根据用户ID获取用户详细信息，包括姓名、手机号、用户的user_id、open_id、union_id等。"""
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        # 创建client
        client = _create_lark_client(final_app_id, final_app_secret)

        from lark_oapi.api.contact.v3 import GetUserRequest, GetUserResponse  # type: ignore[import-untyped]

        # 构造请求对象
        request: GetUserRequest = (
            GetUserRequest.builder().user_id(user_id).user_id_type(user_id_type).department_id_type(department_id_type).build()
        )

        # 发起请求
        response: GetUserResponse = client.contact.v3.user.get(request)

        # 处理失败返回
        if not response.success():
            error_msg = f"获取用户信息失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)

            # 尝试获取详细错误信息
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
            except Exception:
                pass

            return error_msg

        # 处理业务结果，过滤掉avatar字段
        user_data_raw = _marshal_json(response.data, indent=4)
        user_data = cast(dict[str, Any], json.loads(user_data_raw))

        # 移除avatar字段
        if "user" in user_data and "avatar" in user_data["user"]:
            del user_data["user"]["avatar"]

        return json.dumps(user_data, indent=4, ensure_ascii=False)

    except BaseException as e:
        error_msg = f"获取用户信息失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg


def get_user_id_by_name(name: Annotated[str, "要查找的用户姓名"]) -> str:
    """根据用户名返回用户信息。如果遇到重名情况，会返回错误信息和所有重名用户列表。

    参数说明：
    - name: 要查找的用户姓名

    返回：
    - 成功且唯一：返回包含member_id、member_id_type、name的用户信息
    - 重名情况：返回错误信息和所有重名用户的列表
    - 未找到：返回未找到用户的错误信息
    """
    try:
        # 1. 首先获取所有群组
        chat_list_response = get_feishu_chat_list(page_size=100)

        # 检查响应是否是有效的JSON
        try:
            chat_list = cast(dict[str, Any], json.loads(chat_list_response))
        except json.JSONDecodeError:
            return json.dumps(
                {
                    "error": "获取群组列表失败",
                    "message": f"群组列表API返回错误: {chat_list_response}",
                },
                ensure_ascii=False,
            )

        if "items" not in chat_list:
            return json.dumps(
                {
                    "error": "获取群组列表失败",
                    "message": f"群组列表格式异常: {chat_list_response}",
                },
                ensure_ascii=False,
            )

        # 2. 遍历所有群组查找用户，收集所有匹配的用户
        found_users: list[dict[str, str]] = []
        processed_user_ids: set[str] = set()  # 用于去重，避免同一用户在多个群组中被重复添加

        chat_items = cast(list[dict[str, Any]], chat_list["items"])
        for chat in chat_items:
            chat_id = chat.get("chat_id")
            if not chat_id:
                continue

            # 获取群组成员
            members_response = search_users_in_chat(chat_id=chat_id)

            # 检查成员响应是否是有效的JSON
            try:
                members = cast(dict[str, Any], json.loads(members_response))
            except json.JSONDecodeError:
                # 如果某个群组获取失败，跳过继续处理其他群组
                logger.warning(f"跳过群组 {chat_id}，获取成员失败: {members_response}")
                continue

            if "users" in members:
                users = cast(list[dict[str, Any]], members["users"])
                for user in users:
                    if user.get("name") == name:
                        user_id_raw = user.get("member_id")
                        user_id: str = user_id_raw if isinstance(user_id_raw, str) else ""

                        # 避免重复添加同一用户（可能在多个群组中）
                        if user_id not in processed_user_ids:
                            member_id_type_raw = user.get("member_id_type", "open_id")
                            member_id_type: str = member_id_type_raw if isinstance(member_id_type_raw, str) else "open_id"
                            user_name_raw = user.get("name", name)
                            user_name_value: str = user_name_raw if isinstance(user_name_raw, str) else name
                            user_info: dict[str, str] = {
                                "member_id": user_id,
                                "member_id_type": member_id_type,
                                "name": user_name_value,
                            }
                            found_users.append(user_info)
                            processed_user_ids.add(user_id)

        # 3. 根据找到的用户数量返回不同结果
        if len(found_users) == 0:
            return json.dumps(
                {"error": "未找到用户", "message": f"未找到名为 '{name}' 的用户"},
                ensure_ascii=False,
            )

        elif len(found_users) == 1:
            # 只找到一个用户，返回用户信息
            return json.dumps(found_users[0], ensure_ascii=False)

        else:
            # 找到多个用户（重名情况）
            return json.dumps(
                {
                    "error": "发现重名用户",
                    "message": f"发现 {len(found_users)} 个名为 '{name}' 的用户",
                    "duplicate_users": found_users,
                },
                ensure_ascii=False,
            )

    except Exception as e:
        error_msg = f"查找用户信息失败. Error: {repr(e)}"
        logger.error(error_msg)
        return json.dumps(
            {"error": "查找失败", "message": error_msg},
            ensure_ascii=False,
        )


def add_message_reaction(
    message_id: Annotated[str, "消息ID"],
    emoji_type: Annotated[
        str,
        "emoji类型，如：THUMBSUP, THUMBSDOWN, HEART, FIRE, CLAP, THUMBSDOWN, DONE, OneSecond等",
    ],
) -> str:
    """给消息添加表情回应"""
    try:
        # 获取配置
        config_result = validate_feishu_config()
        if isinstance(config_result, str):
            # 配置验证失败，返回错误信息
            return config_result

        final_app_id, final_app_secret = config_result

        client = _create_lark_client(final_app_id, final_app_secret)

        request: CreateMessageReactionRequest = (
            CreateMessageReactionRequest.builder()
            .request_body(
                CreateMessageReactionRequestBody.builder().reaction_type(Emoji.builder().emoji_type(emoji_type).build()).build(),
            )
            .message_id(message_id)
            .build()
        )

        response: CreateMessageReactionResponse = client.im.v1.message_reaction.create(
            request,
        )

        if not response.success():
            error_msg = f"添加消息反应失败, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            logger.error(error_msg)
            try:
                error_detail = json.loads(response.raw.content)
                error_msg += f"\n详细错误: {json.dumps(error_detail, indent=2, ensure_ascii=False)}"
            except Exception:
                pass
            return error_msg

        return f"成功添加表情反应: {emoji_type}"

    except Exception as e:
        error_msg = f"添加消息反应失败. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg
