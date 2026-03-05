import { RFC_FULL_CONTENT } from "./mockEvents";

// ---------------------------------------------------------------------------
// Mock 文件树节点
// ---------------------------------------------------------------------------
export interface MockFileNode {
  name: string;
  path: string;
  isDirectory: boolean;
  isFile: boolean;
  size: number;
  children?: MockFileNode[];
  expanded?: boolean;
}

// ---------------------------------------------------------------------------
// Mock 文件内容
// ---------------------------------------------------------------------------
export interface MockFileContent {
  path: string;
  content: string;
  size: number;
  truncated: boolean;
  language: string;
}

// ---------------------------------------------------------------------------
// 文件树结构
// ---------------------------------------------------------------------------
export const MOCK_FILE_TREE: MockFileNode[] = [
  {
    name: "rfcs",
    path: "rfcs",
    isDirectory: true,
    isFile: false,
    size: 0,
    expanded: true,
    children: [
      {
        name: "0001-jxw-agent.md",
        path: "rfcs/0001-jxw-agent.md",
        isDirectory: false,
        isFile: true,
        size: RFC_FULL_CONTENT.length,
      },
    ],
  },
  {
    name: "skills",
    path: "skills",
    isDirectory: true,
    isFile: false,
    size: 0,
    children: [
      {
        name: "enterprise_query",
        path: "skills/enterprise_query",
        isDirectory: true,
        isFile: false,
        size: 0,
        children: [
          { name: "search_enterprise.yaml", path: "skills/enterprise_query/search_enterprise.yaml", isDirectory: false, isFile: true, size: 320 },
          { name: "get_enterprise_detail.yaml", path: "skills/enterprise_query/get_enterprise_detail.yaml", isDirectory: false, isFile: true, size: 280 },
          { name: "handler.py", path: "skills/enterprise_query/handler.py", isDirectory: false, isFile: true, size: 1540 },
          { name: "SKILL.md", path: "skills/enterprise_query/SKILL.md", isDirectory: false, isFile: true, size: 420 },
        ],
      },
      {
        name: "business_analysis",
        path: "skills/business_analysis",
        isDirectory: true,
        isFile: false,
        size: 0,
        children: [
          { name: "analyze_trend.yaml", path: "skills/business_analysis/analyze_trend.yaml", isDirectory: false, isFile: true, size: 360 },
          { name: "compare_enterprises.yaml", path: "skills/business_analysis/compare_enterprises.yaml", isDirectory: false, isFile: true, size: 310 },
          { name: "handler.py", path: "skills/business_analysis/handler.py", isDirectory: false, isFile: true, size: 1820 },
          { name: "SKILL.md", path: "skills/business_analysis/SKILL.md", isDirectory: false, isFile: true, size: 380 },
        ],
      },
      {
        name: "policy_match",
        path: "skills/policy_match",
        isDirectory: true,
        isFile: false,
        size: 0,
        children: [
          { name: "match_policy.yaml", path: "skills/policy_match/match_policy.yaml", isDirectory: false, isFile: true, size: 340 },
          { name: "list_policies.yaml", path: "skills/policy_match/list_policies.yaml", isDirectory: false, isFile: true, size: 290 },
          { name: "handler.py", path: "skills/policy_match/handler.py", isDirectory: false, isFile: true, size: 1650 },
          { name: "SKILL.md", path: "skills/policy_match/SKILL.md", isDirectory: false, isFile: true, size: 350 },
        ],
      },
      {
        name: "supply_chain",
        path: "skills/supply_chain",
        isDirectory: true,
        isFile: false,
        size: 0,
        children: [
          { name: "get_supply_chain.yaml", path: "skills/supply_chain/get_supply_chain.yaml", isDirectory: false, isFile: true, size: 320 },
          { name: "find_industry_cluster.yaml", path: "skills/supply_chain/find_industry_cluster.yaml", isDirectory: false, isFile: true, size: 350 },
          { name: "handler.py", path: "skills/supply_chain/handler.py", isDirectory: false, isFile: true, size: 1480 },
          { name: "SKILL.md", path: "skills/supply_chain/SKILL.md", isDirectory: false, isFile: true, size: 360 },
        ],
      },
      {
        name: "risk_alert",
        path: "skills/risk_alert",
        isDirectory: true,
        isFile: false,
        size: 0,
        children: [
          { name: "check_risk.yaml", path: "skills/risk_alert/check_risk.yaml", isDirectory: false, isFile: true, size: 310 },
          { name: "scan_abnormal.yaml", path: "skills/risk_alert/scan_abnormal.yaml", isDirectory: false, isFile: true, size: 340 },
          { name: "handler.py", path: "skills/risk_alert/handler.py", isDirectory: false, isFile: true, size: 1720 },
          { name: "SKILL.md", path: "skills/risk_alert/SKILL.md", isDirectory: false, isFile: true, size: 370 },
        ],
      },
    ],
  },
  {
    name: "data",
    path: "data",
    isDirectory: true,
    isFile: false,
    size: 0,
    children: [
      { name: "schema.sql", path: "data/schema.sql", isDirectory: false, isFile: true, size: 890 },
    ],
  },
  { name: "nexau.yaml", path: "nexau.yaml", isDirectory: false, isFile: true, size: 420 },
  { name: "run.py", path: "run.py", isDirectory: false, isFile: true, size: 680 },
  { name: "requirements.txt", path: "requirements.txt", isDirectory: false, isFile: true, size: 120 },
  { name: "Dockerfile", path: "Dockerfile", isDirectory: false, isFile: true, size: 350 },
];

// ---------------------------------------------------------------------------
// 文件内容映射
// ---------------------------------------------------------------------------
export const MOCK_FILE_CONTENTS: Record<string, MockFileContent> = {
  "rfcs/0001-jxw-agent.md": {
    path: "rfcs/0001-jxw-agent.md",
    content: RFC_FULL_CONTENT,
    size: RFC_FULL_CONTENT.length,
    truncated: false,
    language: "markdown",
  },
  "nexau.yaml": {
    path: "nexau.yaml",
    content: `name: jxw-data-analyst
description: 经信委数据分析智能体

llm:
  provider: openai
  model: gpt-4o
  temperature: 0.3

skills:
  - enterprise_query
  - business_analysis
  - policy_match
  - supply_chain
  - risk_alert

env:
  DATABASE_URL: \${DATABASE_URL}
`,
    size: 420,
    truncated: false,
    language: "yaml",
  },
  "data/schema.sql": {
    path: "data/schema.sql",
    content: `CREATE TABLE enterprise_info (
  id SERIAL PRIMARY KEY,
  name VARCHAR(200) NOT NULL,
  industry VARCHAR(100),
  region VARCHAR(100),
  registered_capital DECIMAL(15,2),
  employee_count INTEGER,
  established_date DATE
);

CREATE TABLE financial_data (
  id SERIAL PRIMARY KEY,
  enterprise_id INTEGER REFERENCES enterprise_info(id),
  year INTEGER NOT NULL,
  quarter INTEGER NOT NULL,
  revenue DECIMAL(15,2),
  profit DECIMAL(15,2),
  tax_amount DECIMAL(15,2)
);

CREATE TABLE policy_catalog (
  id SERIAL PRIMARY KEY,
  title VARCHAR(300) NOT NULL,
  category VARCHAR(100),
  target_industry VARCHAR(200),
  min_revenue DECIMAL(15,2),
  min_employees INTEGER,
  effective_date DATE,
  expiry_date DATE
);

CREATE TABLE supply_chain (
  id SERIAL PRIMARY KEY,
  upstream_id INTEGER REFERENCES enterprise_info(id),
  downstream_id INTEGER REFERENCES enterprise_info(id),
  relationship_type VARCHAR(50),
  trade_volume DECIMAL(15,2)
);
`,
    size: 890,
    truncated: false,
    language: "sql",
  },
  "run.py": {
    path: "run.py",
    content: `#!/usr/bin/env python3
"""经信委数据分析智能体 — 启动入口"""

import asyncio
from nexau import Agent

async def main():
    agent = Agent.from_config("nexau.yaml")
    await agent.start()

if __name__ == "__main__":
    asyncio.run(main())
`,
    size: 680,
    truncated: false,
    language: "python",
  },
  "skills/enterprise_query/SKILL.md": {
    path: "skills/enterprise_query/SKILL.md",
    content: `---
name: enterprise_query
description: 按名称、行业、地区等多维度检索企业信息
license: MIT
---

## Tools

| Tool | 说明 |
|------|------|
| search_enterprise | 多维度企业搜索 |
| get_enterprise_detail | 获取企业详细信息 |

## 数据表

- \`enterprise_info\` — 12,847 条企业记录
`,
    size: 420,
    truncated: false,
    language: "markdown",
  },
  "skills/enterprise_query/handler.py": {
    path: "skills/enterprise_query/handler.py",
    content: `"""enterprise_query Skill handler.

RFC-0001: 企业信息检索 Skill

实现按名称/行业/地区多维度企业检索。
"""

import asyncpg
from nexau.skill import SkillHandler, tool


class EnterpriseQueryHandler(SkillHandler):
    """企业信息检索 Skill."""

    async def setup(self):
        self.pool = await asyncpg.create_pool(self.env["DATABASE_URL"])

    @tool
    async def search_enterprise(
        self, keyword: str | None = None,
        industry: str | None = None,
        region: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """按名称/行业/地区搜索企业."""
        query = """
            SELECT id, name, industry, region,
                   registered_capital, employee_count
            FROM enterprise_info
            WHERE ($1 IS NULL OR name ILIKE '%' || $1 || '%')
              AND ($2 IS NULL OR industry = $2)
              AND ($3 IS NULL OR region = $3)
            ORDER BY registered_capital DESC
            LIMIT $4
        """
        rows = await self.pool.fetch(query, keyword, industry, region, limit)
        return [dict(r) for r in rows]

    @tool
    async def get_enterprise_detail(self, enterprise_id: int) -> dict:
        """获取企业详细信息."""
        row = await self.pool.fetchrow(
            "SELECT * FROM enterprise_info WHERE id = $1",
            enterprise_id,
        )
        return dict(row) if row else {}
`,
    size: 1540,
    truncated: false,
    language: "python",
  },
};
