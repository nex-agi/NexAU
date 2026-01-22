"""Filter DSL 模型定义

基于 PostgREST 风格的可序列化 Filter 系统，支持 JSON 序列化、
SQLAlchemy 转换、Python 内存评估和 HTTP 查询字符串序列化。
"""

from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class FilterOperator(str, Enum):
    """PostgREST 风格的过滤操作符"""

    EQ = "eq"  # 等于
    NEQ = "neq"  # 不等于
    GT = "gt"  # 大于
    GTE = "gte"  # 大于等于
    LT = "lt"  # 小于
    LTE = "lte"  # 小于等于
    LIKE = "like"  # 模式匹配（大小写敏感）
    ILIKE = "ilike"  # 模式匹配（大小写不敏感）
    IN = "in"  # 包含于列表
    IS = "is"  # IS NULL / IS NOT NULL


class LogicalOperator(str, Enum):
    """逻辑组合操作符"""

    AND = "and"
    OR = "or"
    NOT = "not"


class FilterBase(BaseModel):
    """Filter DSL 基类"""

    pass


class ComparisonFilter(FilterBase):
    """单字段比较过滤器"""

    type: Literal["comparison"] = "comparison"
    field: str
    op: FilterOperator
    value: str | int | float | bool | None | list[str | int | float]

    @classmethod
    def eq(cls, field: str, value: str | int | float | bool | None) -> "ComparisonFilter":
        """创建等于过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.EQ, value=value)

    @classmethod
    def neq(cls, field: str, value: str | int | float | bool | None) -> "ComparisonFilter":
        """创建不等于过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.NEQ, value=value)

    @classmethod
    def gt(cls, field: str, value: str | int | float) -> "ComparisonFilter":
        """创建大于过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.GT, value=value)

    @classmethod
    def gte(cls, field: str, value: str | int | float) -> "ComparisonFilter":
        """创建大于等于过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.GTE, value=value)

    @classmethod
    def lt(cls, field: str, value: str | int | float) -> "ComparisonFilter":
        """创建小于过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.LT, value=value)

    @classmethod
    def lte(cls, field: str, value: str | int | float) -> "ComparisonFilter":
        """创建小于等于过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.LTE, value=value)

    @classmethod
    def like(cls, field: str, value: str) -> "ComparisonFilter":
        """创建模式匹配过滤器的便捷方法（大小写敏感）"""
        return cls(field=field, op=FilterOperator.LIKE, value=value)

    @classmethod
    def ilike(cls, field: str, value: str) -> "ComparisonFilter":
        """创建模式匹配过滤器的便捷方法（大小写不敏感）"""
        return cls(field=field, op=FilterOperator.ILIKE, value=value)

    @classmethod
    def in_(cls, field: str, value: list[str | int | float]) -> "ComparisonFilter":
        """创建包含于列表过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.IN, value=value)

    @classmethod
    def is_null(cls, field: str) -> "ComparisonFilter":
        """创建 IS NULL 过滤器的便捷方法"""
        return cls(field=field, op=FilterOperator.IS, value=None)


class AndFilter(FilterBase):
    """AND 逻辑组合过滤器"""

    type: Literal["and"] = "and"
    filters: Sequence["ComparisonFilter | AndFilter | OrFilter | NotFilter"]


class OrFilter(FilterBase):
    """OR 逻辑组合过滤器"""

    type: Literal["or"] = "or"
    filters: Sequence["ComparisonFilter | AndFilter | OrFilter | NotFilter"]


class NotFilter(FilterBase):
    """NOT 逻辑取反过滤器"""

    type: Literal["not"] = "not"
    filter: "ComparisonFilter | AndFilter | OrFilter | NotFilter"


# 统一的 Filter 类型别名
Filter = ComparisonFilter | AndFilter | OrFilter | NotFilter

# 解决前向引用，使 Pydantic 能够正确处理嵌套的 Filter 类型
# 这对于支持多态反序列化（discriminated union）是必需的
AndFilter.model_rebuild()
OrFilter.model_rebuild()
NotFilter.model_rebuild()
