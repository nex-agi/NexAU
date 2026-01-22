"""Filter Converter 实现

将 Filter DSL 转换为 SQLAlchemy ColumnElement、Python 内存评估和 HTTP 查询字符串。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from pydantic import BaseModel
from sqlalchemy import ColumnElement, and_, not_, or_
from sqlmodel import SQLModel

from .dsl import (
    AndFilter,
    ComparisonFilter,
    Filter,
    FilterOperator,
    NotFilter,
    OrFilter,
)

if TYPE_CHECKING:
    pass


def to_sqlalchemy(
    filter_: Filter,
    model_class: type[SQLModel],
) -> ColumnElement[bool]:
    """将 Filter DSL 转换为 SQLAlchemy ColumnElement.

    Args:
        filter_: Filter DSL 实例
        model_class: SQLModel 类，用于获取列定义

    Returns:
        SQLAlchemy ColumnElement[bool] 表达式

    Raises:
        ValueError: 如果字段名在 model_class 中不存在

    Examples:
        >>> from sqlmodel import SQLModel, Field
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        ...     age: int
        >>> filter_ = ComparisonFilter.eq("name", "alice")
        >>> expr = to_sqlalchemy(filter_, User)
        >>> # expr is equivalent to User.name == "alice"
    """
    if isinstance(filter_, ComparisonFilter):
        return _convert_comparison_filter(filter_, model_class)
    if isinstance(filter_, AndFilter):
        return _convert_and_filter(filter_, model_class)
    if isinstance(filter_, OrFilter):
        return _convert_or_filter(filter_, model_class)
    # At this point, filter_ must be NotFilter based on the Filter union type
    return _convert_not_filter(filter_, model_class)


def _get_column(
    model_class: type[SQLModel],
    field_name: str,
) -> ColumnElement[object]:
    """获取 model_class 中指定字段的列对象.

    Args:
        model_class: SQLModel 类
        field_name: 字段名

    Returns:
        SQLAlchemy Column 对象

    Raises:
        ValueError: 如果字段名在 model_class 中不存在
    """
    # 检查字段是否存在于模型中
    if not hasattr(model_class, field_name):
        raise ValueError(f"Field '{field_name}' not found in model {model_class.__name__}")

    column: ColumnElement[object] = getattr(model_class, field_name)
    return column


def _convert_comparison_filter(
    filter_: ComparisonFilter,
    model_class: type[SQLModel],
) -> ColumnElement[bool]:
    """将 ComparisonFilter 转换为 SQLAlchemy 表达式.

    Args:
        filter_: ComparisonFilter 实例
        model_class: SQLModel 类

    Returns:
        SQLAlchemy ColumnElement[bool] 表达式
    """
    column = _get_column(model_class, filter_.field)
    op = filter_.op
    value = filter_.value

    if op == FilterOperator.EQ:
        return column == value
    elif op == FilterOperator.NEQ:
        return column != value
    elif op == FilterOperator.GT:
        return column > value
    elif op == FilterOperator.GTE:
        return column >= value
    elif op == FilterOperator.LT:
        return column < value
    elif op == FilterOperator.LTE:
        return column <= value
    elif op == FilterOperator.LIKE:
        return column.like(value)
    elif op == FilterOperator.ILIKE:
        return column.ilike(value)
    elif op == FilterOperator.IN:
        if not isinstance(value, list):
            raise ValueError(f"Invalid value type for operator {op}: expected list, got {type(value).__name__}")
        return column.in_(value)
    elif op == FilterOperator.IS:
        # IS operator is used for NULL checks
        # value should be None for IS NULL
        return column.is_(value)
    else:
        raise ValueError(f"Unsupported operator: {op}")


def _convert_and_filter(
    filter_: AndFilter,
    model_class: type[SQLModel],
) -> ColumnElement[bool]:
    """将 AndFilter 转换为 SQLAlchemy AND 表达式.

    Args:
        filter_: AndFilter 实例
        model_class: SQLModel 类

    Returns:
        SQLAlchemy ColumnElement[bool] 表达式
    """
    if not filter_.filters:
        # Empty AND filter should return True (identity element for AND)
        # Using SQLAlchemy's literal_column for a true expression
        from sqlalchemy import literal

        return literal(True)

    sub_expressions = [to_sqlalchemy(sub_filter, model_class) for sub_filter in filter_.filters]
    return and_(*sub_expressions)


def _convert_or_filter(
    filter_: OrFilter,
    model_class: type[SQLModel],
) -> ColumnElement[bool]:
    """将 OrFilter 转换为 SQLAlchemy OR 表达式.

    Args:
        filter_: OrFilter 实例
        model_class: SQLModel 类

    Returns:
        SQLAlchemy ColumnElement[bool] 表达式
    """
    if not filter_.filters:
        # Empty OR filter should return False (identity element for OR)
        from sqlalchemy import literal

        return literal(False)

    sub_expressions = [to_sqlalchemy(sub_filter, model_class) for sub_filter in filter_.filters]
    return or_(*sub_expressions)


def _convert_not_filter(
    filter_: NotFilter,
    model_class: type[SQLModel],
) -> ColumnElement[bool]:
    """将 NotFilter 转换为 SQLAlchemy NOT 表达式.

    Args:
        filter_: NotFilter 实例
        model_class: SQLModel 类

    Returns:
        SQLAlchemy ColumnElement[bool] 表达式
    """
    sub_expression = to_sqlalchemy(filter_.filter, model_class)
    return not_(sub_expression)


# ============================================================================
# Python 内存评估 (Requirement 3.x)
# ============================================================================


def evaluate(
    filter_: Filter,
    record: Mapping[str, Any] | BaseModel,
) -> bool:
    """在 Python 中评估 Filter DSL.

    Args:
        filter_: Filter DSL 实例
        record: 要评估的记录，可以是字典或 Pydantic/SQLModel 实例

    Returns:
        布尔值，表示记录是否匹配过滤器

    Examples:
        >>> filter_ = ComparisonFilter.eq("name", "alice")
        >>> evaluate(filter_, {"name": "alice", "age": 25})
        True
        >>> evaluate(filter_, {"name": "bob", "age": 30})
        False

        >>> # 也支持 Pydantic model
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> evaluate(filter_, User(name="alice", age=25))
        True
    """
    # 如果是 Pydantic model，转换为字典
    record_dict: Mapping[str, Any]
    if isinstance(record, BaseModel):
        record_dict = record.model_dump()
    else:
        record_dict = record

    if isinstance(filter_, ComparisonFilter):
        return _evaluate_comparison_filter(filter_, record_dict)
    if isinstance(filter_, AndFilter):
        return _evaluate_and_filter(filter_, record_dict)
    if isinstance(filter_, OrFilter):
        return _evaluate_or_filter(filter_, record_dict)
    # At this point, filter_ must be NotFilter based on the Filter union type
    return _evaluate_not_filter(filter_, record_dict)


def _get_field_value(
    record: Mapping[str, Any],
    field_name: str,
) -> Any:
    """获取记录中指定字段的值.

    如果字段不存在，返回 None（需求 3.12）。

    Args:
        record: 记录字典
        field_name: 字段名

    Returns:
        字段值，如果字段不存在则返回 None
    """
    return record.get(field_name, None)


def _safe_compare(a: object, b: object, op: str) -> bool:
    """安全地比较两个值.

    Args:
        a: 左操作数
        b: 右操作数
        op: 比较操作符 ('gt', 'gte', 'lt', 'lte')

    Returns:
        比较结果，如果类型不可比较则返回 False
    """
    try:
        if op == "gt":
            return a > b  # type: ignore[operator]
        elif op == "gte":
            return a >= b  # type: ignore[operator]
        elif op == "lt":
            return a < b  # type: ignore[operator]
        elif op == "lte":
            return a <= b  # type: ignore[operator]
        return False
    except TypeError:
        return False


def _convert_like_pattern_to_regex(pattern: str) -> str:
    """将 SQL LIKE 模式转换为正则表达式.

    SQL LIKE 通配符:
    - % 匹配任意数量的字符（包括零个）
    - _ 匹配单个字符

    Args:
        pattern: SQL LIKE 模式字符串

    Returns:
        等价的正则表达式字符串
    """
    # 首先转义所有正则表达式特殊字符
    # 但保留 % 和 _ 用于后续转换
    result = ""
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "%":
            result += ".*"
        elif char == "_":
            result += "."
        elif char in r"\^$.|?*+()[]{}":
            # 转义正则表达式特殊字符
            result += "\\" + char
        else:
            result += char
        i += 1

    # 添加锚点以确保完整匹配
    return "^" + result + "$"


def _evaluate_comparison_filter(
    filter_: ComparisonFilter,
    record: Mapping[str, Any],
) -> bool:
    """评估 ComparisonFilter.

    Args:
        filter_: ComparisonFilter 实例
        record: 记录字典

    Returns:
        布尔值，表示记录是否匹配过滤器
    """
    field_value = _get_field_value(record, filter_.field)
    op = filter_.op
    filter_value = filter_.value

    if op == FilterOperator.EQ:
        # 需求 3.2: eq 操作符返回 record[field] == value
        return field_value == filter_value

    elif op == FilterOperator.NEQ:
        # 需求 3.3: neq 操作符返回 record[field] != value
        return field_value != filter_value

    elif op == FilterOperator.GT:
        # 需求 3.4: gt 操作符返回 record[field] > value
        if field_value is None or filter_value is None:
            return False
        return _safe_compare(field_value, filter_value, "gt")

    elif op == FilterOperator.GTE:
        # 需求 3.4: gte 操作符返回 record[field] >= value
        if field_value is None or filter_value is None:
            return False
        return _safe_compare(field_value, filter_value, "gte")

    elif op == FilterOperator.LT:
        # 需求 3.4: lt 操作符返回 record[field] < value
        if field_value is None or filter_value is None:
            return False
        return _safe_compare(field_value, filter_value, "lt")

    elif op == FilterOperator.LTE:
        # 需求 3.4: lte 操作符返回 record[field] <= value
        if field_value is None or filter_value is None:
            return False
        return _safe_compare(field_value, filter_value, "lte")

    elif op == FilterOperator.LIKE:
        # 需求 3.5: like 操作符使用通配符模式匹配（% 匹配任意字符）
        if field_value is None or filter_value is None:
            return False
        if not isinstance(field_value, str) or not isinstance(filter_value, str):
            return False
        regex_pattern = _convert_like_pattern_to_regex(filter_value)
        return bool(re.match(regex_pattern, field_value))

    elif op == FilterOperator.ILIKE:
        # 需求 3.6: ilike 操作符使用大小写不敏感的通配符模式匹配
        if field_value is None or filter_value is None:
            return False
        if not isinstance(field_value, str) or not isinstance(filter_value, str):
            return False
        regex_pattern = _convert_like_pattern_to_regex(filter_value)
        return bool(re.match(regex_pattern, field_value, re.IGNORECASE))

    elif op == FilterOperator.IN:
        # 需求 3.7: in 操作符返回 record[field] in value
        if not isinstance(filter_value, list):
            raise ValueError(f"Invalid value type for operator {op}: expected list, got {type(filter_value).__name__}")
        return field_value in filter_value

    elif op == FilterOperator.IS:
        # 需求 3.8: is 操作符且值为 null 返回 record[field] is None
        # IS operator is used for NULL checks
        if filter_value is None:
            return field_value is None
        else:
            # IS with non-null value (e.g., IS TRUE, IS FALSE)
            return field_value is filter_value

    else:
        raise ValueError(f"Unsupported operator: {op}")


def _evaluate_and_filter(
    filter_: AndFilter,
    record: Mapping[str, Any],
) -> bool:
    """评估 AndFilter.

    需求 3.9: and 逻辑返回所有子过滤器的逻辑与结果。

    Args:
        filter_: AndFilter 实例
        record: 记录字典

    Returns:
        布尔值，表示记录是否匹配所有子过滤器
    """
    if not filter_.filters:
        # 空 AND 过滤器返回 True（AND 的恒等元素）
        return True

    return all(evaluate(sub_filter, record) for sub_filter in filter_.filters)


def _evaluate_or_filter(
    filter_: OrFilter,
    record: Mapping[str, Any],
) -> bool:
    """评估 OrFilter.

    需求 3.10: or 逻辑返回所有子过滤器的逻辑或结果。

    Args:
        filter_: OrFilter 实例
        record: 记录字典

    Returns:
        布尔值，表示记录是否匹配任一子过滤器
    """
    if not filter_.filters:
        # 空 OR 过滤器返回 False（OR 的恒等元素）
        return False

    return any(evaluate(sub_filter, record) for sub_filter in filter_.filters)


def _evaluate_not_filter(
    filter_: NotFilter,
    record: Mapping[str, Any],
) -> bool:
    """评估 NotFilter.

    需求 3.11: not 逻辑返回子过滤器的逻辑非结果。

    Args:
        filter_: NotFilter 实例
        record: 记录字典

    Returns:
        布尔值，表示记录是否不匹配子过滤器
    """
    return not evaluate(filter_.filter, record)


# ============================================================================
# HTTP 查询字符串序列化 (Requirement 4.x)
# ============================================================================


def _url_encode_single_value(value: str | int | float | bool | list[str | int | float] | None) -> str:
    """URL 编码单个值（非列表）.

    Args:
        value: 要编码的值

    Returns:
        URL 编码后的字符串
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        # 列表值不应该直接编码，应该使用 IN 操作符的特殊格式
        raise ValueError("List values should be encoded using IN operator format")
    # 对字符串值进行 URL 编码
    return quote(str(value), safe="")


def _format_filter_for_nested(filter_: Filter) -> str:
    """将 Filter 格式化为嵌套格式（用于 and/or 内部）.

    嵌套格式: field.op.value（不带等号）

    Args:
        filter_: Filter DSL 实例

    Returns:
        嵌套格式的字符串
    """
    if isinstance(filter_, ComparisonFilter):
        return _format_comparison_filter_nested(filter_)
    if isinstance(filter_, AndFilter):
        return _format_and_filter_nested(filter_)
    if isinstance(filter_, OrFilter):
        return _format_or_filter_nested(filter_)
    # At this point, filter_ must be NotFilter based on the Filter union type
    return _format_not_filter_nested(filter_)


def _format_comparison_filter_nested(filter_: ComparisonFilter) -> str:
    """将 ComparisonFilter 格式化为嵌套格式.

    格式: field.op.value

    Args:
        filter_: ComparisonFilter 实例

    Returns:
        嵌套格式的字符串
    """
    field = filter_.field
    op = filter_.op.value
    value = filter_.value

    if filter_.op == FilterOperator.IN:
        # 需求 4.3: in 操作符格式为 field.in.(v1,v2,...)
        if not isinstance(value, list):
            raise ValueError(f"Invalid value type for operator {op}: expected list, got {type(value).__name__}")
        encoded_values = [_url_encode_single_value(v) for v in value]
        return f"{field}.{op}.({','.join(encoded_values)})"
    else:
        # 需求 4.2: 其他操作符格式为 field.op.value
        encoded_value = _url_encode_single_value(value)
        return f"{field}.{op}.{encoded_value}"


def _format_and_filter_nested(filter_: AndFilter) -> str:
    """将 AndFilter 格式化为嵌套格式.

    格式: and(f1,f2,...)

    Args:
        filter_: AndFilter 实例

    Returns:
        嵌套格式的字符串
    """
    if not filter_.filters:
        return "and()"

    nested_filters = [_format_filter_for_nested(f) for f in filter_.filters]
    return f"and({','.join(nested_filters)})"


def _format_or_filter_nested(filter_: OrFilter) -> str:
    """将 OrFilter 格式化为嵌套格式.

    格式: or(f1,f2,...)

    Args:
        filter_: OrFilter 实例

    Returns:
        嵌套格式的字符串
    """
    if not filter_.filters:
        return "or()"

    nested_filters = [_format_filter_for_nested(f) for f in filter_.filters]
    return f"or({','.join(nested_filters)})"


def _format_not_filter_nested(filter_: NotFilter) -> str:
    """将 NotFilter 格式化为嵌套格式.

    需求 4.6: not 逻辑格式为 not.filter

    Args:
        filter_: NotFilter 实例

    Returns:
        嵌套格式的字符串
    """
    inner = filter_.filter

    if isinstance(inner, ComparisonFilter):
        # 对于 ComparisonFilter，格式为 field.not.op.value
        field = inner.field
        op = inner.op.value
        value = inner.value

        if inner.op == FilterOperator.IN:
            if not isinstance(value, list):
                raise ValueError(f"Invalid value type for operator {op}: expected list, got {type(value).__name__}")
            encoded_values = [_url_encode_single_value(v) for v in value]
            return f"{field}.not.{op}.({','.join(encoded_values)})"
        else:
            encoded_value = _url_encode_single_value(value)
            return f"{field}.not.{op}.{encoded_value}"
    else:
        # 对于逻辑过滤器，格式为 not.filter
        inner_str = _format_filter_for_nested(inner)
        return f"not.{inner_str}"


def to_query_string(filter_: Filter) -> str:
    """将 Filter DSL 序列化为 PostgREST 风格的查询字符串.

    需求 4.1: 提供 to_query_string(filter: Filter_DSL) -> str 方法

    Args:
        filter_: Filter DSL 实例

    Returns:
        PostgREST 风格的查询字符串

    Examples:
        >>> filter_ = ComparisonFilter.eq("name", "alice")
        >>> to_query_string(filter_)
        'name=eq.alice'

        >>> filter_ = ComparisonFilter.in_("status", ["active", "pending"])
        >>> to_query_string(filter_)
        'status=in.(active,pending)'

        >>> filter_ = AndFilter(filters=[ComparisonFilter.gte("age", 18), ComparisonFilter.eq("status", "active")])
        >>> to_query_string(filter_)
        'and=(age.gte.18,status.eq.active)'

        >>> filter_ = NotFilter(filter=ComparisonFilter.eq("status", "deleted"))
        >>> to_query_string(filter_)
        'status=not.eq.deleted'
    """
    if isinstance(filter_, ComparisonFilter):
        return _to_query_string_comparison(filter_)
    if isinstance(filter_, AndFilter):
        return _to_query_string_and(filter_)
    if isinstance(filter_, OrFilter):
        return _to_query_string_or(filter_)
    # At this point, filter_ must be NotFilter based on the Filter union type
    return _to_query_string_not(filter_)


def _to_query_string_comparison(filter_: ComparisonFilter) -> str:
    """将 ComparisonFilter 序列化为查询字符串.

    需求 4.2: ComparisonFilter 序列化为 "field=op.value" 格式
    需求 4.3: in 操作符序列化为 "field=in.(value1,value2,...)" 格式

    Args:
        filter_: ComparisonFilter 实例

    Returns:
        查询字符串
    """
    field = filter_.field
    op = filter_.op.value
    value = filter_.value

    if filter_.op == FilterOperator.IN:
        # 需求 4.3: in 操作符格式为 field=in.(v1,v2,...)
        if not isinstance(value, list):
            raise ValueError(f"Invalid value type for operator {op}: expected list, got {type(value).__name__}")
        encoded_values = [_url_encode_single_value(v) for v in value]
        return f"{field}={op}.({','.join(encoded_values)})"
    else:
        # 需求 4.2: 其他操作符格式为 field=op.value
        encoded_value = _url_encode_single_value(value)
        return f"{field}={op}.{encoded_value}"


def _to_query_string_and(filter_: AndFilter) -> str:
    """将 AndFilter 序列化为查询字符串.

    需求 4.4: AndFilter 序列化为 "and=(filter1,filter2,...)" 格式

    Args:
        filter_: AndFilter 实例

    Returns:
        查询字符串
    """
    if not filter_.filters:
        return "and=()"

    nested_filters = [_format_filter_for_nested(f) for f in filter_.filters]
    return f"and=({','.join(nested_filters)})"


def _to_query_string_or(filter_: OrFilter) -> str:
    """将 OrFilter 序列化为查询字符串.

    需求 4.5: OrFilter 序列化为 "or=(filter1,filter2,...)" 格式

    Args:
        filter_: OrFilter 实例

    Returns:
        查询字符串
    """
    if not filter_.filters:
        return "or=()"

    nested_filters = [_format_filter_for_nested(f) for f in filter_.filters]
    return f"or=({','.join(nested_filters)})"


def _to_query_string_not(filter_: NotFilter) -> str:
    """将 NotFilter 序列化为查询字符串.

    需求 4.6: NotFilter 序列化为 "not.filter" 格式
    对于 ComparisonFilter，格式为 "field=not.op.value"

    Args:
        filter_: NotFilter 实例

    Returns:
        查询字符串
    """
    inner = filter_.filter

    if isinstance(inner, ComparisonFilter):
        # 对于 ComparisonFilter，格式为 field=not.op.value
        field = inner.field
        op = inner.op.value
        value = inner.value

        if inner.op == FilterOperator.IN:
            if not isinstance(value, list):
                raise ValueError(f"Invalid value type for operator {op}: expected list, got {type(value).__name__}")
            encoded_values = [_url_encode_single_value(v) for v in value]
            return f"{field}=not.{op}.({','.join(encoded_values)})"
        else:
            encoded_value = _url_encode_single_value(value)
            return f"{field}=not.{op}.{encoded_value}"
    else:
        # 对于逻辑过滤器，格式为 not.filter
        inner_str = _format_filter_for_nested(inner)
        return f"not={inner_str}"


# ============================================================================
# HTTP 查询字符串反序列化 (Requirement 4.8)
# ============================================================================


# 有效的操作符集合
_VALID_OPERATORS = {op.value for op in FilterOperator}


def _url_decode_value(encoded: str) -> str | int | float | bool | None:
    """URL 解码并转换值类型.

    Args:
        encoded: URL 编码的字符串值

    Returns:
        解码并转换类型后的值
    """
    # URL 解码
    decoded = unquote(encoded)

    # 处理特殊值
    if decoded == "null":
        return None
    if decoded == "true":
        return True
    if decoded == "false":
        return False

    # 尝试转换为数字
    try:
        # 尝试整数
        if "." not in decoded and "e" not in decoded.lower():
            return int(decoded)
    except ValueError:
        pass

    try:
        # 尝试浮点数
        return float(decoded)
    except ValueError:
        pass

    # 返回字符串
    return decoded


def _split_by_comma_at_depth_zero(s: str) -> list[str]:
    """在深度为 0 的位置按逗号分割字符串.

    只在括号外的逗号处分割，保持括号内的内容完整。

    Args:
        s: 要分割的字符串

    Returns:
        分割后的字符串列表
    """
    result: list[str] = []
    current: list[str] = []
    depth = 0

    for char in s:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        result.append("".join(current))

    return result


def _parse_nested_filter(nested: str) -> Filter:
    """解析嵌套格式的过滤器.

    嵌套格式示例:
    - field.op.value (ComparisonFilter)
    - field.op.(v1,v2,...) (ComparisonFilter with IN)
    - field.not.op.value (NotFilter with ComparisonFilter)
    - and(f1,f2,...) (AndFilter)
    - or(f1,f2,...) (OrFilter)
    - not.filter (NotFilter)

    Args:
        nested: 嵌套格式的字符串

    Returns:
        解析后的 Filter 实例

    Raises:
        ValueError: 如果格式无效
    """
    nested = nested.strip()

    if not nested:
        raise ValueError("Invalid query string format: empty filter")

    # 处理 and(...) 格式
    if nested.startswith("and(") and nested.endswith(")"):
        inner = nested[4:-1]  # 去掉 "and(" 和 ")"
        if not inner:
            return AndFilter(filters=[])
        parts = _split_by_comma_at_depth_zero(inner)
        filters = [_parse_nested_filter(p) for p in parts]
        return AndFilter(filters=filters)

    # 处理 or(...) 格式
    if nested.startswith("or(") and nested.endswith(")"):
        inner = nested[3:-1]  # 去掉 "or(" 和 ")"
        if not inner:
            return OrFilter(filters=[])
        parts = _split_by_comma_at_depth_zero(inner)
        filters = [_parse_nested_filter(p) for p in parts]
        return OrFilter(filters=filters)

    # 处理 not.filter 格式（逻辑过滤器的 NOT）
    if nested.startswith("not."):
        inner = nested[4:]  # 去掉 "not."
        inner_filter = _parse_nested_filter(inner)
        return NotFilter(filter=inner_filter)

    # 处理 field.not.op.value 格式（ComparisonFilter 的 NOT）
    # 和 field.op.value 格式
    return _parse_comparison_nested(nested)


def _parse_comparison_nested(nested: str) -> Filter:
    """解析嵌套格式的 ComparisonFilter.

    格式:
    - field.op.value
    - field.op.(v1,v2,...)
    - field.not.op.value
    - field.not.op.(v1,v2,...)

    Args:
        nested: 嵌套格式的字符串

    Returns:
        解析后的 Filter 实例

    Raises:
        ValueError: 如果格式无效
    """
    # 找到第一个点的位置（字段名后面）
    first_dot = nested.find(".")
    if first_dot == -1:
        raise ValueError(f"Invalid query string format: {nested}")

    field = nested[:first_dot]
    rest = nested[first_dot + 1 :]

    # 检查是否是 NOT 格式
    is_not = False
    if rest.startswith("not."):
        is_not = True
        rest = rest[4:]  # 去掉 "not."

    # 找到操作符
    second_dot = rest.find(".")
    if second_dot == -1:
        raise ValueError(f"Invalid query string format: {nested}")

    op_str = rest[:second_dot]
    value_str = rest[second_dot + 1 :]

    # 验证操作符
    if op_str not in _VALID_OPERATORS:
        raise ValueError(f"Unknown operator in query string: {op_str}")

    op = FilterOperator(op_str)

    # 解析值
    if op == FilterOperator.IN:
        # IN 操作符的值格式为 (v1,v2,...)
        if not value_str.startswith("(") or not value_str.endswith(")"):
            raise ValueError(f"Invalid query string format: {nested}")
        inner = value_str[1:-1]  # 去掉括号
        if not inner:
            values: list[str | int | float] = []
        else:
            parts = inner.split(",")
            values = []
            for p in parts:
                decoded = _url_decode_value(p)
                # IN 操作符只支持 str, int, float 类型
                if decoded is None or isinstance(decoded, bool):
                    values.append(str(decoded) if decoded is not None else "null")
                else:
                    values.append(decoded)
        comparison = ComparisonFilter(field=field, op=op, value=values)
    else:
        value = _url_decode_value(value_str)
        comparison = ComparisonFilter(field=field, op=op, value=value)

    if is_not:
        return NotFilter(filter=comparison)
    return comparison


def from_query_string(query: str) -> Filter:
    """从 PostgREST 风格的查询字符串解析 Filter DSL.

    需求 4.8: 提供 from_query_string(query: str) -> Filter_DSL 方法进行反向解析

    支持的格式:
    - field=op.value (ComparisonFilter)
    - field=in.(v1,v2,...) (ComparisonFilter with IN)
    - field=not.op.value (NotFilter with ComparisonFilter)
    - and=(f1,f2,...) (AndFilter)
    - or=(f1,f2,...) (OrFilter)
    - not=filter (NotFilter with logical filter)

    Args:
        query: PostgREST 风格的查询字符串

    Returns:
        解析后的 Filter DSL 实例

    Raises:
        ValueError: 如果查询字符串格式无效、操作符不识别或括号不匹配

    Examples:
        >>> from_query_string("name=eq.alice")
        ComparisonFilter(type='comparison', field='name', op=<FilterOperator.EQ: 'eq'>, value='alice')

        >>> from_query_string("status=in.(active,pending)")
        ComparisonFilter(type='comparison', field='status', op=<FilterOperator.IN: 'in'>, value=['active', 'pending'])

        >>> from_query_string("and=(age.gte.18,status.eq.active)")
        AndFilter(type='and', filters=[...])

        >>> from_query_string("status=not.eq.deleted")
        NotFilter(type='not', filter=ComparisonFilter(...))
    """
    query = query.strip()

    if not query:
        raise ValueError("Invalid query string format: empty query")

    # 找到等号位置
    eq_pos = query.find("=")
    if eq_pos == -1:
        raise ValueError(f"Invalid query string format: {query}")

    left = query[:eq_pos]
    right = query[eq_pos + 1 :]

    # 处理 and=(...) 格式
    if left == "and":
        if right.startswith("("):
            if not right.endswith(")"):
                raise ValueError(f"Invalid query string format: {query}")
            inner = right[1:-1]  # 去掉括号
            if not inner:
                return AndFilter(filters=[])
            parts = _split_by_comma_at_depth_zero(inner)
            filters = [_parse_nested_filter(p) for p in parts]
            return AndFilter(filters=filters)
        if not any(right.startswith(f"{op}.") for op in _VALID_OPERATORS):
            raise ValueError(f"Invalid query string format: {query}")

    # 处理 or=(...) 格式
    if left == "or":
        if right.startswith("("):
            if not right.endswith(")"):
                raise ValueError(f"Invalid query string format: {query}")
            inner = right[1:-1]  # 去掉括号
            if not inner:
                return OrFilter(filters=[])
            parts = _split_by_comma_at_depth_zero(inner)
            filters = [_parse_nested_filter(p) for p in parts]
            return OrFilter(filters=filters)
        if not any(right.startswith(f"{op}.") for op in _VALID_OPERATORS):
            raise ValueError(f"Invalid query string format: {query}")

    # 处理 not=filter 格式（逻辑过滤器的 NOT）
    if left == "not":
        try:
            inner_filter = _parse_nested_filter(right)
        except ValueError:
            inner_filter = None
        if inner_filter is not None:
            return NotFilter(filter=inner_filter)

    # 处理 field=op.value 或 field=not.op.value 格式
    field = left

    # 检查是否是 NOT 格式
    is_not = False
    if right.startswith("not."):
        is_not = True
        right = right[4:]  # 去掉 "not."

    # 找到操作符
    dot_pos = right.find(".")
    if dot_pos == -1:
        raise ValueError(f"Invalid query string format: {query}")

    op_str = right[:dot_pos]
    value_str = right[dot_pos + 1 :]

    # 验证操作符
    if op_str not in _VALID_OPERATORS:
        raise ValueError(f"Unknown operator in query string: {op_str}")

    op = FilterOperator(op_str)

    # 解析值
    if op == FilterOperator.IN:
        # IN 操作符的值格式为 (v1,v2,...)
        if not value_str.startswith("(") or not value_str.endswith(")"):
            raise ValueError(f"Invalid query string format: {query}")
        inner = value_str[1:-1]  # 去掉括号
        if not inner:
            values_list: list[str | int | float] = []
        else:
            parts = inner.split(",")
            values_list = []
            for p in parts:
                decoded = _url_decode_value(p)
                # IN 操作符只支持 str, int, float 类型
                if decoded is None or isinstance(decoded, bool):
                    values_list.append(str(decoded) if decoded is not None else "null")
                else:
                    values_list.append(decoded)
        comparison = ComparisonFilter(field=field, op=op, value=values_list)
    else:
        value = _url_decode_value(value_str)
        comparison = ComparisonFilter(field=field, op=op, value=value)

    if is_not:
        return NotFilter(filter=comparison)
    return comparison
