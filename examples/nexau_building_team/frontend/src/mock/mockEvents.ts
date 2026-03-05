import type { TeamStreamEnvelope, AgentEvent, TaskInfo } from "../types";
import {
  TEAM_ID, RUN_ID,
  LEADER_ID, LEADER_ROLE,
  RFC_WRITER_ID, RFC_WRITER_ROLE,
  BUILDER_DB_ID, BUILDER_DB_ROLE,
  BUILDER_SKILL_ID, BUILDER_SKILL_ROLE,
  BUILDER_CFG_ID, BUILDER_CFG_ROLE,
  Phase,
} from "./mockConstants";
import { IMPL_TASK_SNAPSHOTS } from "./mockTasks";

// ═══════════════════════════════════════════════════════════════════════════
//  RFC 文档内容（完整版，含 Mermaid 和多数据库 Skill）
// ═══════════════════════════════════════════════════════════════════════════
export const RFC_FULL_CONTENT = `# RFC-0001: 经信委数据分析智能体

- **状态**: draft
- **优先级**: P0
- **标签**: \`data-analysis\`, \`government\`, \`multi-skill\`
- **影响服务**: nexau-agent, skill-runtime
- **创建日期**: 2026-03-02

## 摘要

基于经信委 PostgreSQL 数据库，构建一个具备企业信息检索、经营数据分析、政策智能匹配、产业链图谱分析和风险预警能力的 NexAU 智能体。

## 系统架构

\`\`\`mermaid
graph TB
    User([用户]) -->|自然语言| Agent[经信委数据分析智能体]
    Agent -->|调用| S1[enterprise_query Skill]
    Agent -->|调用| S2[business_analysis Skill]
    Agent -->|调用| S3[policy_match Skill]
    Agent -->|调用| S4[supply_chain Skill]
    Agent -->|调用| S5[risk_alert Skill]

    S1 -->|SQL| DB1[(enterprise_info<br/>12,847 条)]
    S2 -->|SQL| DB2[(financial_data<br/>营收/利润)]
    S3 -->|SQL| DB3[(policy_catalog<br/>政策库)]
    S4 -->|SQL| DB4[(supply_chain<br/>产业链)]
    S5 -->|SQL| DB2
    S5 -->|SQL| DB1

    style Agent fill:#6366f1,color:#fff
    style S1 fill:#2563eb,color:#fff
    style S2 fill:#2563eb,color:#fff
    style S3 fill:#2563eb,color:#fff
    style S4 fill:#2563eb,color:#fff
    style S5 fill:#2563eb,color:#fff
\`\`\`

## 数据库 ER 关系

\`\`\`mermaid
erDiagram
    enterprise_info ||--o{ financial_data : "1:N"
    enterprise_info ||--o{ supply_chain : "upstream"
    enterprise_info ||--o{ supply_chain : "downstream"
    enterprise_info }o--o{ policy_catalog : "匹配"

    enterprise_info {
        int id PK
        varchar name
        varchar industry
        varchar region
        decimal registered_capital
        int employee_count
        date established_date
    }

    financial_data {
        int id PK
        int enterprise_id FK
        int year
        int quarter
        decimal revenue
        decimal profit
        decimal tax_amount
    }

    policy_catalog {
        int id PK
        varchar title
        varchar category
        varchar target_industry
        decimal min_revenue
        int min_employees
        date effective_date
        date expiry_date
    }

    supply_chain {
        int id PK
        int upstream_id FK
        int downstream_id FK
        varchar relationship_type
        decimal trade_volume
    }
\`\`\`

## Skills 详细设计

### Skill 1: enterprise_query — 企业信息检索

| 属性 | 说明 |
|------|------|
| **功能** | 按名称、行业、地区等多维度检索企业信息 |
| **数据表** | enterprise_info |
| **Tools** | search_enterprise, get_enterprise_detail |

**输入参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| keyword | string | 否 | 企业名称关键词 |
| industry | string | 否 | 行业筛选 |
| region | string | 否 | 地区筛选 |
| limit | integer | 否 | 返回数量限制 (默认 20) |

**SQL 模板:**
\`\`\`sql
SELECT id, name, industry, region, registered_capital,
       employee_count, established_date
FROM enterprise_info
WHERE ($1 IS NULL OR name ILIKE '%' || $1 || '%')
  AND ($2 IS NULL OR industry = $2)
  AND ($3 IS NULL OR region = $3)
ORDER BY registered_capital DESC
LIMIT $4;
\`\`\`

### Skill 2: business_analysis — 经营数据分析

| 属性 | 说明 |
|------|------|
| **功能** | 分析企业营收、利润、税收的年度/季度趋势 |
| **数据表** | financial_data JOIN enterprise_info |
| **Tools** | analyze_trend, compare_enterprises |

**输入参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| enterprise_id | integer | 是 | 企业 ID |
| start_year | integer | 否 | 起始年份 (默认 2020) |
| end_year | integer | 否 | 截止年份 (默认 2025) |
| metrics | string[] | 否 | 指标: revenue, profit, tax_amount |

**SQL 模板:**
\`\`\`sql
SELECT year, quarter,
       SUM(revenue) as total_revenue,
       SUM(profit) as total_profit,
       ROUND(SUM(profit)/NULLIF(SUM(revenue),0)*100, 2) as profit_margin
FROM financial_data
WHERE enterprise_id = $1
  AND year BETWEEN $2 AND $3
GROUP BY year, quarter
ORDER BY year, quarter;
\`\`\`

### Skill 3: policy_match — 政策智能匹配

| 属性 | 说明 |
|------|------|
| **功能** | 根据企业特征（行业、规模、营收）智能匹配适用政策 |
| **数据表** | policy_catalog JOIN enterprise_info |
| **Tools** | match_policy, list_policies |

**匹配流程:**

\`\`\`mermaid
flowchart LR
    A[输入企业ID] --> B[查询企业信息]
    B --> C{行业匹配}
    C -->|匹配| D{营收门槛}
    C -->|不匹配| G[排除]
    D -->|达标| E{员工规模}
    D -->|未达标| G
    E -->|达标| F[✓ 纳入匹配结果]
    E -->|未达标| G
    F --> H[按匹配度排序输出]
\`\`\`

### Skill 4: supply_chain — 产业链分析

| 属性 | 说明 |
|------|------|
| **功能** | 分析企业上下游关系、产业集群、贸易链路 |
| **数据表** | supply_chain JOIN enterprise_info |
| **Tools** | get_supply_chain, find_industry_cluster |

**关系图谱查询:**
\`\`\`sql
-- 获取企业的直接上下游
SELECT sc.relationship_type, sc.trade_volume,
       e_up.name as upstream_name, e_up.industry as upstream_industry,
       e_down.name as downstream_name, e_down.industry as downstream_industry
FROM supply_chain sc
JOIN enterprise_info e_up ON sc.upstream_id = e_up.id
JOIN enterprise_info e_down ON sc.downstream_id = e_down.id
WHERE sc.upstream_id = $1 OR sc.downstream_id = $1
ORDER BY sc.trade_volume DESC;
\`\`\`

### Skill 5: risk_alert — 风险预警

| 属性 | 说明 |
|------|------|
| **功能** | 监测企业经营异常指标，生成风险预警 |
| **数据表** | financial_data JOIN enterprise_info |
| **Tools** | check_risk, scan_abnormal |

**预警规则:**

| 风险类型 | 触发条件 | 等级 |
|----------|----------|------|
| 营收骤降 | 同比下降 > 30% | 高 |
| 利润转负 | 连续 2 季度亏损 | 高 |
| 税收异常 | 税收/营收比偏离行业均值 > 2σ | 中 |
| 规模萎缩 | 员工数同比减少 > 20% | 中 |

## 智能体配置

\`\`\`yaml
name: jxw-data-analyst
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
\`\`\`

## 实现计划

| 阶段 | 任务 | 负责 |
|------|------|------|
| Phase 1 | 数据库探索 + enterprise_query Skill | Builder-DB |
| Phase 2 | business_analysis + policy_match Skill | Builder-Skill |
| Phase 3 | supply_chain + risk_alert Skill | Builder-Skill |
| Phase 4 | 智能体配置 + 启动脚本 | Builder-Cfg |
| Phase 5 | 集成测试 | Builder-DB |
`;

// ---------------------------------------------------------------------------
// TimedEnvelope — 带延迟和控制标记的事件包装
// ---------------------------------------------------------------------------
export interface TimedEnvelope {
  envelope: TeamStreamEnvelope;
  delay: number;
  /** 发出此事件后暂停回放，等待 sendUserMessage 恢复 */
  pauseAfter?: boolean;
  /** 发出此事件后更新 mockTasks 为此快照 */
  taskSnapshot?: TaskInfo[];
}

// ---------------------------------------------------------------------------
// Helper: 构造 TeamStreamEnvelope
// ---------------------------------------------------------------------------
let _msgCounter = 0;
let _tcCounter = 0;

function nextMsgId(): string { return `msg-${++_msgCounter}`; }
function nextTcId(): string { return `tc-${++_tcCounter}`; }

function env(
  agentId: string,
  roleName: string,
  event: AgentEvent,
  delay: number,
  extra?: Partial<Pick<TimedEnvelope, "pauseAfter" | "taskSnapshot">>,
): TimedEnvelope {
  return {
    envelope: {
      team_id: TEAM_ID,
      agent_id: agentId,
      role_name: roleName,
      run_id: RUN_ID,
      event,
    },
    delay,
    ...extra,
  };
}

// ---------------------------------------------------------------------------
// Helper: 将文本拆成 TEXT_MESSAGE_CONTENT delta 序列
// ---------------------------------------------------------------------------
function streamText(
  agentId: string,
  role: string,
  text: string,
  opts?: { chunkSize?: number; delay?: number; startDelay?: number },
): TimedEnvelope[] {
  const { chunkSize = 4, delay = 25, startDelay = 300 } = opts ?? {};
  const events: TimedEnvelope[] = [];
  const msgId = nextMsgId();
  events.push(env(agentId, role, { type: "TEXT_MESSAGE_START", message_id: msgId }, startDelay));
  for (let i = 0; i < text.length; i += chunkSize) {
    events.push(env(agentId, role, {
      type: "TEXT_MESSAGE_CONTENT", message_id: msgId, delta: text.slice(i, i + chunkSize),
    }, delay));
  }
  events.push(env(agentId, role, { type: "TEXT_MESSAGE_END", message_id: msgId }, 50));
  return events;
}

// ---------------------------------------------------------------------------
// Helper: 将思考文本拆成 THINKING delta 序列
// ---------------------------------------------------------------------------
function streamThinking(
  agentId: string,
  role: string,
  text: string,
  opts?: { chunkSize?: number; delay?: number; startDelay?: number },
): TimedEnvelope[] {
  const { chunkSize = 6, delay = 30, startDelay = 400 } = opts ?? {};
  const events: TimedEnvelope[] = [];
  const msgId = nextMsgId();
  events.push(env(agentId, role, {
    type: "THINKING_TEXT_MESSAGE_START", message_id: msgId,
  }, startDelay));
  for (let i = 0; i < text.length; i += chunkSize) {
    events.push(env(agentId, role, {
      type: "THINKING_TEXT_MESSAGE_CONTENT", delta: text.slice(i, i + chunkSize),
    }, delay));
  }
  events.push(env(agentId, role, { type: "THINKING_TEXT_MESSAGE_END" }, 50));
  return events;
}

// ---------------------------------------------------------------------------
// Helper: 完整 tool call 序列 (START → ARGS → END → RESULT)
// ---------------------------------------------------------------------------
function toolCall(
  agentId: string,
  role: string,
  name: string,
  args: string,
  result: string,
  opts?: {
    startDelay?: number;
    argsChunkSize?: number;
    argsDelay?: number;
    resultDelay?: number;
    extra?: Partial<Pick<TimedEnvelope, "pauseAfter" | "taskSnapshot">>;
  },
): TimedEnvelope[] {
  const {
    startDelay = 400, argsChunkSize = 60,
    argsDelay = 35, resultDelay = 200, extra,
  } = opts ?? {};
  const events: TimedEnvelope[] = [];
  const tcId = nextTcId();
  const msgId = nextMsgId();

  events.push(env(agentId, role, {
    type: "TOOL_CALL_START", tool_call_id: tcId,
    tool_call_name: name, name, message_id: msgId,
  }, startDelay));

  for (let i = 0; i < args.length; i += argsChunkSize) {
    events.push(env(agentId, role, {
      type: "TOOL_CALL_ARGS", tool_call_id: tcId, delta: args.slice(i, i + argsChunkSize),
    }, argsDelay));
  }

  // TOOL_CALL_END
  events.push(env(agentId, role, {
    type: "TOOL_CALL_END", tool_call_id: tcId,
  }, 50));

  // TOOL_CALL_RESULT — pauseAfter / taskSnapshot 标记在最后一个事件上
  events.push(env(agentId, role, {
    type: "TOOL_CALL_RESULT", tool_call_id: tcId, content: result,
  }, resultDelay, extra));

  return events;
}

// ---------------------------------------------------------------------------
// Helper: TEAM_MESSAGE 事件
// ---------------------------------------------------------------------------
function teamMessage(
  agentId: string,
  role: string,
  fromAgentId: string,
  content: string,
  delay = 300,
): TimedEnvelope {
  return env(agentId, role, {
    type: "TEAM_MESSAGE", content, from_agent_id: fromAgentId, to_agent_id: agentId,
  }, delay);
}

// ---------------------------------------------------------------------------
// Helper: RUN_STARTED / RUN_FINISHED
// ---------------------------------------------------------------------------
function runStarted(agentId: string, role: string, delay = 200): TimedEnvelope {
  return env(agentId, role, {
    type: "RUN_STARTED", agent_id: agentId, run_id: RUN_ID,
  }, delay);
}

function runFinished(agentId: string, role: string, delay = 300): TimedEnvelope {
  return env(agentId, role, {
    type: "RUN_FINISHED", run_id: RUN_ID, message: "done",
  }, delay);
}

// ---------------------------------------------------------------------------
// Helper: 交错合并多个事件流
// ---------------------------------------------------------------------------
function interleaveEvents(...streams: TimedEnvelope[][]): TimedEnvelope[] {
  const result: TimedEnvelope[] = [];
  const indices = streams.map(() => 0);
  const batchSize = 4;

  let safety = 0;
  while (indices.some((idx, i) => idx < streams[i].length)) {
    if (++safety > 10000) break;
    for (let s = 0; s < streams.length; s++) {
      const count = Math.min(batchSize, streams[s].length - indices[s]);
      for (let b = 0; b < count; b++) {
        result.push(streams[s][indices[s]]);
        indices[s]++;
      }
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Phase 0: REQUIREMENTS — Leader 分析需求
// ═══════════════════════════════════════════════════════════════════════════
function buildRequirementsPhase(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];

  // 1. RUN_STARTED
  events.push(runStarted(LEADER_ID, LEADER_ROLE, 100));

  // 2. Leader 思考
  events.push(...streamThinking(LEADER_ID, LEADER_ROLE,
    "用户希望基于经信委数据库构建一个 NexAU 智能体，需要支持企业分析和政策分析。\n\n" +
    "我需要了解以下关键信息：\n" +
    "1. 数据库类型和连接方式\n" +
    "2. 用户需要的具体分析类型\n" +
    "3. 是否有现成的数据字典或表结构文档\n\n" +
    "先创建任务列表，然后向用户确认需求细节。",
  ));

  // 3. Leader 输出文字
  events.push(...streamText(LEADER_ID, LEADER_ROLE,
    "好的，我来分析您的需求。您希望基于经信委的数据库，构建一个能够进行企业分析和政策分析的 NexAU 智能体。\n\n" +
    "我将按以下步骤完成构建：\n\n" +
    "1. **需求确认** — 确认数据库结构和分析需求范围\n" +
    "2. **设计文档** — 编写智能体设计方案 (RFC)\n" +
    "3. **Skills 开发** — 实现各类分析能力\n" +
    "4. **配置与脚本** — 编写智能体配置和启动脚本\n" +
    "5. **测试验收** — 运行端到端测试\n\n" +
    "首先，我需要向您确认几个关键问题：",
  ));

  // 4. write_todos
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "确认数据库结构和分析需求", status: "in_progress" },
        { description: "创建 RFC 设计文档", status: "pending" },
        { description: "开发分析 Skills", status: "pending" },
        { description: "编写配置和启动脚本", status: "pending" },
        { description: "测试验收", status: "pending" },
      ],
    }),
    "Todos updated",
    { startDelay: 500 },
  ));

  // 5. ask_user — 需求澄清（pauseAfter 在 TOOL_CALL_END 上）
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "ask_user",
    JSON.stringify({
      questions: [
        {
          header: "数据库类型",
          question: "经信委数据库使用的是什么类型的数据库系统？",
          type: "choice",
          options: [
            { label: "PostgreSQL", description: "开源关系型数据库，支持丰富的数据类型" },
            { label: "MySQL", description: "流行的开源关系型数据库" },
            { label: "SQL Server", description: "微软关系型数据库" },
          ],
        },
        {
          header: "分析需求",
          question: "您希望智能体支持哪些类型的分析能力？（可多选）",
          type: "choice",
          multiSelect: true,
          options: [
            { label: "企业基本信息查询", description: "按名称、行业、地区多维度检索企业信息" },
            { label: "经营数据分析", description: "营收、利润、增长趋势等财务分析" },
            { label: "政策匹配分析", description: "根据企业特征智能匹配适用政策" },
            { label: "产业链分析", description: "上下游企业关系、产业集群分析" },
          ],
        },
        {
          header: "数据字典",
          question: "您是否有数据库的数据字典或表结构文档？这将帮助我更准确地设计查询逻辑。",
          type: "yesno",
        },
      ],
    }),
    "Waiting for user response...",
    { startDelay: 600, extra: { pauseAfter: true } },
  ));

  return events;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Phase 2: DESIGN — Leader 创建任务 + spawn RFC Writer
// ═══════════════════════════════════════════════════════════════════════════
function buildDesignPhase(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];

  // 1. Leader 收到用户回答后的思考
  events.push(...streamThinking(LEADER_ID, LEADER_ROLE,
    "用户确认使用 PostgreSQL 数据库，需要企业信息查询、经营数据分析、政策匹配分析和产业链分析四项能力。" +
    "有数据字典可用。\n\n" +
    "接下来我需要：\n" +
    "1. 创建 RFC 设计任务\n" +
    "2. Spawn RFC Writer Agent 来编写设计文档\n" +
    "3. 等待设计方案审批后再进入实施阶段",
    { startDelay: 600 },
  ));

  // 2. Leader 更新 todos
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "确认数据库结构和分析需求", status: "completed" },
        { description: "创建 RFC 设计文档", status: "in_progress" },
        { description: "开发分析 Skills", status: "pending" },
        { description: "编写配置和启动脚本", status: "pending" },
        { description: "测试验收", status: "pending" },
      ],
    }),
    "Todos updated",
    { startDelay: 300 },
  ));

  // 3. Leader 输出文字
  events.push(...streamText(LEADER_ID, LEADER_ROLE,
    "感谢您的确认！需求已明确：\n\n" +
    "- **数据库**: PostgreSQL\n" +
    "- **分析能力**: 企业查询、经营分析、政策匹配、产业链分析\n" +
    "- **数据字典**: 可用\n\n" +
    "我正在创建 RFC Writer Agent 来编写详细的设计方案...",
  ));

  // 4. spawn_teammate (rfc_writer)
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "spawn_teammate",
    JSON.stringify({ role: "rfc_writer" }),
    JSON.stringify({ agent_id: RFC_WRITER_ID, role_name: RFC_WRITER_ROLE, status: "running" }),
    { startDelay: 500 },
  ));

  // 5. create_task (RFC-001)
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "create_task",
    JSON.stringify({
      title: "经信委智能体设计文档",
      description: "编写智能体架构和 Skills 设计方案 (RFC-0001)",
      priority: 1,
    }),
    JSON.stringify({ task_id: "RFC-001" }),
  ));

  // 6. claim_task
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "claim_task",
    JSON.stringify({ task_id: "RFC-001", assignee_agent_id: RFC_WRITER_ID }),
    "Task RFC-001 claimed by rfc-writer-001",
  ));

  // 7. message → RFC Writer
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "message",
    JSON.stringify({
      content: "请为经信委数据分析智能体编写 RFC 设计文档。数据库类型：PostgreSQL。需要支持的分析能力：企业信息查询、经营数据分析、政策匹配、产业链分析。有数据字典可用。请详细设计每个 Skill 的输入输出和数据库查询逻辑。",
      to_agent_id: RFC_WRITER_ID,
    }),
    "Message sent to rfc-writer-001",
  ));

  // ── RFC Writer 开始工作 ──────────────────────────────────────────

  // 8. RUN_STARTED for RFC Writer
  events.push(runStarted(RFC_WRITER_ID, RFC_WRITER_ROLE, 500));

  // 9. RFC Writer 接收 team message
  events.push(teamMessage(RFC_WRITER_ID, RFC_WRITER_ROLE, LEADER_ID,
    "请为经信委数据分析智能体编写 RFC 设计文档。数据库类型：PostgreSQL。需要支持的分析能力：企业信息查询、经营数据分析、政策匹配、产业链分析。",
    200,
  ));

  // 10. RFC Writer 思考
  events.push(...streamThinking(RFC_WRITER_ID, RFC_WRITER_ROLE,
    "收到任务，需要为经信委数据分析智能体编写 RFC 设计文档。\n\n" +
    "我需要设计以下 Skills：\n" +
    "1. enterprise_query — 企业信息多维度检索\n" +
    "2. business_analysis — 经营数据趋势分析\n" +
    "3. policy_match — 根据企业特征匹配政策\n" +
    "4. supply_chain — 产业链上下游关系分析\n" +
    "5. risk_alert — 经营异常风险预警\n\n" +
    "需要包含 Mermaid 架构图、ER 关系图，以及每个 Skill 的 SQL 模板。\n" +
    "让我先查看项目结构和数据字典，然后编写完整的 RFC 文档。",
  ));

  // 11. RFC Writer 的 write_todos
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "分析数据库表结构", status: "in_progress" },
        { description: "设计 enterprise_query Skill", status: "pending" },
        { description: "设计 business_analysis Skill", status: "pending" },
        { description: "设计 policy_match Skill", status: "pending" },
        { description: "设计 supply_chain Skill", status: "pending" },
        { description: "设计 risk_alert Skill", status: "pending" },
        { description: "编写 RFC 文档 (含 Mermaid 架构图)", status: "pending" },
        { description: "提交审批", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 12. RFC Writer 探索项目 (list_directory)
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "list_directory",
    JSON.stringify({ path: "." }),
    JSON.stringify({
      files: ["nexau.yaml", "run.py", "requirements.txt"],
      dirs: ["skills/", "rfcs/", "tools/", "data/"],
    }),
  ));

  // 13. RFC Writer 读取数据字典
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "read_file",
    JSON.stringify({ path: "data/schema.sql" }),
    "CREATE TABLE enterprise_info (\n  id SERIAL PRIMARY KEY,\n  name VARCHAR(200),\n  industry VARCHAR(100),\n  region VARCHAR(100),\n  registered_capital DECIMAL,\n  employee_count INTEGER,\n  established_date DATE\n);\n\nCREATE TABLE financial_data (\n  id SERIAL PRIMARY KEY,\n  enterprise_id INTEGER REFERENCES enterprise_info(id),\n  year INTEGER,\n  quarter INTEGER,\n  revenue DECIMAL,\n  profit DECIMAL,\n  tax_amount DECIMAL\n);\n\nCREATE TABLE policy_catalog (\n  id SERIAL PRIMARY KEY,\n  title VARCHAR(300),\n  category VARCHAR(100),\n  target_industry VARCHAR(200),\n  min_revenue DECIMAL,\n  min_employees INTEGER,\n  effective_date DATE,\n  expiry_date DATE\n);\n\nCREATE TABLE supply_chain (\n  id SERIAL PRIMARY KEY,\n  upstream_id INTEGER REFERENCES enterprise_info(id),\n  downstream_id INTEGER REFERENCES enterprise_info(id),\n  relationship_type VARCHAR(50),\n  trade_volume DECIMAL\n);",
    { argsChunkSize: 80 },
  ));

  // 14. RFC Writer 更新 todos
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "分析数据库表结构", status: "completed" },
        { description: "设计 enterprise_query Skill", status: "completed" },
        { description: "设计 business_analysis Skill", status: "completed" },
        { description: "设计 policy_match Skill", status: "completed" },
        { description: "设计 supply_chain Skill", status: "completed" },
        { description: "设计 risk_alert Skill", status: "completed" },
        { description: "编写 RFC 文档 (含 Mermaid 架构图)", status: "in_progress" },
        { description: "提交审批", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 15. RFC Writer 写 RFC 文档（使用完整 RFC 内容，含 Mermaid 和多 Skill 设计）
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "write_file",
    JSON.stringify({
      file_path: "rfcs/0001-jxw-agent.md",
      content: RFC_FULL_CONTENT,
    }),
    "File written: rfcs/0001-jxw-agent.md (8.7KB)",
    { argsChunkSize: 120 },
  ));

  // 16. RFC Writer 输出摘要
  events.push(...streamText(RFC_WRITER_ID, RFC_WRITER_ROLE,
    "我已完成设计文档 `rfcs/0001-jxw-agent.md`，包含完整的系统架构图、数据库 ER 图和各 Skill 详细设计。\n\n" +
    "### 智能体设计概要\n\n" +
    "**名称**: 经信委数据分析智能体\n\n" +
    "**数据库**: PostgreSQL (4 张核心表: enterprise_info, financial_data, policy_catalog, supply_chain)\n\n" +
    "| Skill | 功能 | 核心数据表 |\n" +
    "|-------|------|------------|\n" +
    "| enterprise_query | 按名称/行业/地区检索企业 | enterprise_info |\n" +
    "| business_analysis | 营收/利润趋势分析 | financial_data |\n" +
    "| policy_match | 根据企业特征匹配政策 | policy_catalog |\n" +
    "| supply_chain | 上下游企业关系图谱 | supply_chain |\n" +
    "| risk_alert | 经营异常风险预警 | financial_data + enterprise_info |\n\n" +
    "文档中包含 **Mermaid 架构图**、**ER 关系图**、**政策匹配流程图**，以及每个 Skill 的 SQL 查询模板和预警规则。\n\n" +
    "请确认此方案是否满足您的需求。",
    { startDelay: 500 },
  ));

  // 17. RFC Writer 更新 todos
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "分析数据库表结构", status: "completed" },
        { description: "设计 enterprise_query Skill", status: "completed" },
        { description: "设计 business_analysis Skill", status: "completed" },
        { description: "设计 policy_match Skill", status: "completed" },
        { description: "设计 supply_chain Skill", status: "completed" },
        { description: "设计 risk_alert Skill", status: "completed" },
        { description: "编写 RFC 文档 (含 Mermaid 架构图)", status: "completed" },
        { description: "提交审批", status: "in_progress" },
      ],
    }),
    "Todos updated",
  ));

  // 18. RFC Writer ask_user — 设计审批（pauseAfter）
  events.push(...toolCall(RFC_WRITER_ID, RFC_WRITER_ROLE, "ask_user",
    JSON.stringify({
      questions: [
        {
          header: "设计方案审批",
          question: "RFC-0001 设计文档已完成，包含以下 **5 个 Skills** 的详细设计：\n\n" +
            "- **enterprise_query** — 企业信息多维度检索\n" +
            "- **business_analysis** — 经营数据趋势分析\n" +
            "- **policy_match** — 智能政策匹配\n" +
            "- **supply_chain** — 产业链关系图谱\n" +
            "- **risk_alert** — 经营异常风险预警\n\n" +
            "文档包含系统架构 Mermaid 图、数据库 ER 图、政策匹配流程图、SQL 查询模板和预警规则。\n\n" +
            "是否批准此方案并开始实施？",
          type: "yesno",
        },
      ],
    }),
    "Waiting for user approval...",
    { startDelay: 500, extra: { pauseAfter: true } },
  ));

  return events;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Phase 4: IMPLEMENTATION — 3 Builders 并行实施
// ═══════════════════════════════════════════════════════════════════════════
function buildImplementationPhase(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];

  // ── Leader: 分解任务并 spawn builders ──────────────────────────

  // 1. RFC Writer 完成 + Leader 思考
  events.push(runFinished(RFC_WRITER_ID, RFC_WRITER_ROLE, 300));

  events.push(...streamThinking(LEADER_ID, LEADER_ROLE,
    "设计方案已通过审批。现在需要：\n" +
    "1. 将 RFC 分解为具体实施任务\n" +
    "2. Spawn 多个 Builder Agent 并行实施\n" +
    "3. 分配任务：\n" +
    "   - Builder-DB: 探索数据库 + enterprise_query Skill\n" +
    "   - Builder-Skill: business_analysis + policy_match Skills\n" +
    "   - Builder-Cfg: 智能体配置 + 启动脚本",
    { startDelay: 800 },
  ));

  // 2. Leader 更新 todos
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "确认数据库结构和分析需求", status: "completed" },
        { description: "创建 RFC 设计文档", status: "completed" },
        { description: "开发分析 Skills", status: "in_progress" },
        { description: "编写配置和启动脚本", status: "pending" },
        { description: "测试验收", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 3. Leader 输出文字
  events.push(...streamText(LEADER_ID, LEADER_ROLE,
    "设计方案已审批通过！现在开始并行实施。\n\n" +
    "我将创建 3 个 Builder Agent 分工协作：\n\n" +
    "- **Builder-DB** → 探索数据库结构 + 企业查询 Skill\n" +
    "- **Builder-Skill** → 经营分析 + 政策匹配 Skill\n" +
    "- **Builder-Cfg** → 智能体配置 + 启动脚本",
  ));

  // 4. spawn 3 builders
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "spawn_teammate",
    JSON.stringify({ role: "builder" }),
    JSON.stringify({ agent_id: BUILDER_DB_ID, role_name: BUILDER_DB_ROLE, status: "running" }),
    { startDelay: 400 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "spawn_teammate",
    JSON.stringify({ role: "builder" }),
    JSON.stringify({ agent_id: BUILDER_SKILL_ID, role_name: BUILDER_SKILL_ROLE, status: "running" }),
    { startDelay: 200 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "spawn_teammate",
    JSON.stringify({ role: "builder" }),
    JSON.stringify({ agent_id: BUILDER_CFG_ID, role_name: BUILDER_CFG_ROLE, status: "running" }),
    { startDelay: 200 },
  ));

  // 5. 创建实施任务
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "create_task",
    JSON.stringify({ title: "企业信息查询 Skill", description: "实现按名称/行业/地区多维度企业检索", priority: 1 }),
    JSON.stringify({ task_id: "SKILL-001" }),
    { startDelay: 300 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "create_task",
    JSON.stringify({ title: "经营分析 Skill", description: "实现营收、利润、增长趋势分析", priority: 1 }),
    JSON.stringify({ task_id: "SKILL-002" }),
    { startDelay: 100 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "create_task",
    JSON.stringify({ title: "政策匹配 Skill", description: "根据企业特征智能匹配适用政策", priority: 2, dependencies: ["SKILL-002"] }),
    JSON.stringify({ task_id: "SKILL-003" }),
    { startDelay: 100 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "create_task",
    JSON.stringify({ title: "智能体配置与启动脚本", description: "编写 nexau.yaml 和 run.py", priority: 1 }),
    JSON.stringify({ task_id: "CONFIG-001" }),
    { startDelay: 100 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "create_task",
    JSON.stringify({ title: "启动脚本与入口", description: "编写 run.py 和 Dockerfile", priority: 2, dependencies: ["CONFIG-001"] }),
    JSON.stringify({ task_id: "BUILD-001" }),
    { startDelay: 100 },
  ));

  // 6. 分配任务 (message to builders)
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "message",
    JSON.stringify({ content: "SKILL-001: 请探索 PostgreSQL 数据库结构，实现企业信息查询 Skill。数据字典在 data/schema.sql。", to_agent_id: BUILDER_DB_ID }),
    "Message sent",
    { startDelay: 300 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "message",
    JSON.stringify({ content: "SKILL-002 + SKILL-003: 请实现经营分析和政策匹配两个 Skills。参考 rfcs/0001-jxw-agent.md。", to_agent_id: BUILDER_SKILL_ID }),
    "Message sent",
    { startDelay: 100 },
  ));
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "message",
    JSON.stringify({ content: "CONFIG-001 + BUILD-001: 请编写智能体 nexau.yaml 配置和 run.py 启动脚本。参考 RFC 中的 Skills 设计。", to_agent_id: BUILDER_CFG_ID }),
    "Message sent",
    { startDelay: 100 },
  ));

  // ── 3 Builders 并行工作（交错合并） ────────────────────────────
  const builderDbEvents = buildBuilderDbEvents();
  const builderSkillEvents = buildBuilderSkillEvents();
  const builderCfgEvents = buildBuilderCfgEvents();

  events.push(...interleaveEvents(builderDbEvents, builderSkillEvents, builderCfgEvents));

  // ── 测试验收 ─────────────────────────────────────────────────
  events.push(...buildTestingEvents());

  // ── Leader 总结 ──────────────────────────────────────────────
  events.push(...streamText(LEADER_ID, LEADER_ROLE,
    "所有任务已完成！以下是构建结果汇总：\n\n" +
    "### 经信委数据分析智能体 — 构建完成 ✓\n\n" +
    "| 组件 | 状态 | 文件 |\n" +
    "|------|------|------|\n" +
    "| enterprise_query Skill | ✓ 完成 | skills/enterprise_query/ |\n" +
    "| business_analysis Skill | ✓ 完成 | skills/business_analysis/ |\n" +
    "| policy_match Skill | ✓ 完成 | skills/policy_match/ |\n" +
    "| 智能体配置 | ✓ 完成 | nexau.yaml |\n" +
    "| 启动脚本 | ✓ 完成 | run.py |\n" +
    "| 集成测试 | ✓ 全部通过 | — |\n\n" +
    "您可以通过运行 `python run.py` 启动智能体。",
    { startDelay: 800 },
  ));

  // Leader 更新最终 todos
  events.push(...toolCall(LEADER_ID, LEADER_ROLE, "write_todos",
    JSON.stringify({
      todos: [
        { description: "确认数据库结构和分析需求", status: "completed" },
        { description: "创建 RFC 设计文档", status: "completed" },
        { description: "开发分析 Skills", status: "completed" },
        { description: "编写配置和启动脚本", status: "completed" },
        { description: "测试验收", status: "completed" },
      ],
    }),
    "Todos updated",
  ));

  // Leader finish
  events.push(runFinished(LEADER_ID, LEADER_ROLE, 500));

  return events;
}

// ---------------------------------------------------------------------------
// Builder-DB: 探索数据库 + 企业查询 Skill
// ---------------------------------------------------------------------------
function buildBuilderDbEvents(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];
  const A = BUILDER_DB_ID;
  const R = BUILDER_DB_ROLE;

  events.push(runStarted(A, R, 400));

  events.push(teamMessage(A, R, LEADER_ID,
    "SKILL-001: 请探索 PostgreSQL 数据库结构，实现企业信息查询 Skill。", 200));

  events.push(...streamThinking(A, R,
    "我负责探索经信委数据库结构并实现企业查询 Skill。\n" +
    "首先连接数据库，查看表结构和字段信息。",
  ));

  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "连接数据库，探索表结构", status: "in_progress" },
        { description: "分析 enterprise_info 表字段", status: "pending" },
        { description: "编写查询 Tool 定义 (YAML)", status: "pending" },
        { description: "编写查询 Tool 绑定 (Python)", status: "pending" },
        { description: "单元测试", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 探索数据库
  events.push(...toolCall(A, R, "run_shell_command",
    JSON.stringify({ command: "psql -h localhost -U nexau -d jxw_db -c \"\\\\dt\"" }),
    " Schema |      Name       | Type  | Owner\n--------+-----------------+-------+-------\n public | enterprise_info | table | nexau\n public | financial_data  | table | nexau\n public | policy_catalog  | table | nexau\n public | supply_chain    | table | nexau\n(4 rows)",
  ));

  events.push(...toolCall(A, R, "run_shell_command",
    JSON.stringify({ command: "psql -h localhost -U nexau -d jxw_db -c \"\\\\d enterprise_info\"" }),
    "     Column       |          Type          | Nullable\n------------------+------------------------+----------\n id               | integer                | not null\n name             | character varying(200) |\n industry         | character varying(100) |\n region           | character varying(100) |\n registered_capital| numeric               |\n employee_count   | integer                |\n established_date | date                   |",
  ));

  events.push(...toolCall(A, R, "run_shell_command",
    JSON.stringify({ command: "psql -h localhost -U nexau -d jxw_db -c \"SELECT count(*) FROM enterprise_info\"" }),
    " count\n-------\n 12847\n(1 row)",
  ));

  // 更新 todos
  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "连接数据库，探索表结构", status: "completed" },
        { description: "分析 enterprise_info 表字段", status: "completed" },
        { description: "编写查询 Tool 定义 (YAML)", status: "in_progress" },
        { description: "编写查询 Tool 绑定 (Python)", status: "pending" },
        { description: "单元测试", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 写 Tool 定义 YAML
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "skills/enterprise_query/search_enterprise.yaml",
      content: "name: search_enterprise\ndescription: 按名称、行业、地区搜索企业信息\nparameters:\n  type: object\n  properties:\n    keyword:\n      type: string\n      description: 搜索关键词（企业名称）\n    industry:\n      type: string\n      description: 行业筛选\n    region:\n      type: string\n      description: 地区筛选\n    limit:\n      type: integer\n      default: 20\n",
    }),
    "File written: skills/enterprise_query/search_enterprise.yaml",
    { argsChunkSize: 80 },
  ));

  // 写 Tool 绑定 Python
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "skills/enterprise_query/search_enterprise.py",
      content: "import asyncpg\nfrom typing import Optional\n\nasync def search_enterprise(\n    keyword: Optional[str] = None,\n    industry: Optional[str] = None,\n    region: Optional[str] = None,\n    limit: int = 20,\n) -> list[dict]:\n    \"\"\"Search enterprises by name, industry, region.\"\"\"\n    conn = await asyncpg.connect(dsn=os.environ['DATABASE_URL'])\n    conditions = []\n    params = []\n    if keyword:\n        conditions.append(f\"name ILIKE ${{len(params)+1}}\")\n        params.append(f\"%{keyword}%\")\n    if industry:\n        conditions.append(f\"industry = ${{len(params)+1}}\")\n        params.append(industry)\n    if region:\n        conditions.append(f\"region = ${{len(params)+1}}\")\n        params.append(region)\n    where = ' AND '.join(conditions) if conditions else '1=1'\n    rows = await conn.fetch(\n        f\"SELECT * FROM enterprise_info WHERE {where} LIMIT ${len(params)+1}\",\n        *params, limit\n    )\n    await conn.close()\n    return [dict(r) for r in rows]\n",
    }),
    "File written: skills/enterprise_query/search_enterprise.py",
    { argsChunkSize: 100 },
  ));

  // 更新 todos + task snapshot
  const finalTodos = toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "连接数据库，探索表结构", status: "completed" },
        { description: "分析 enterprise_info 表字段", status: "completed" },
        { description: "编写查询 Tool 定义 (YAML)", status: "completed" },
        { description: "编写查询 Tool 绑定 (Python)", status: "completed" },
        { description: "单元测试", status: "completed" },
      ],
    }),
    "Todos updated",
    { startDelay: 300, extra: { taskSnapshot: IMPL_TASK_SNAPSHOTS[0] } },
  );
  events.push(...finalTodos);

  events.push(...streamText(A, R,
    "enterprise_query Skill 实现完成：\n" +
    "- `search_enterprise.yaml` — Tool 定义\n" +
    "- `search_enterprise.py` — 查询实现（支持按名称/行业/地区筛选）\n" +
    "- 数据库共 12,847 条企业记录",
    { startDelay: 200 },
  ));

  events.push(runFinished(A, R, 300));
  return events;
}

// ---------------------------------------------------------------------------
// Builder-Skill: 经营分析 + 政策匹配 Skills
// ---------------------------------------------------------------------------
function buildBuilderSkillEvents(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];
  const A = BUILDER_SKILL_ID;
  const R = BUILDER_SKILL_ROLE;

  events.push(runStarted(A, R, 500));

  events.push(teamMessage(A, R, LEADER_ID,
    "SKILL-002 + SKILL-003: 请实现经营分析和政策匹配两个 Skills。", 200));

  events.push(...streamThinking(A, R,
    "我负责实现经营分析和政策匹配两个 Skills。\n" +
    "先阅读 RFC 设计文档了解详细规格。",
  ));

  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "阅读 RFC 设计文档", status: "in_progress" },
        { description: "实现 business_analysis Skill", status: "pending" },
        { description: "实现 policy_match Skill", status: "pending" },
        { description: "编写测试用例", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 读 RFC
  events.push(...toolCall(A, R, "read_file",
    JSON.stringify({ path: "rfcs/0001-jxw-agent.md" }),
    "# RFC-0001: 经信委数据分析智能体\n\n" +
    "## Skills 详细设计\n\n" +
    "### Skill 2: business_analysis — 经营数据分析\n" +
    "| 属性 | 说明 |\n|------|------|\n| **功能** | 分析企业营收、利润、税收的年度/季度趋势 |\n| **数据表** | financial_data JOIN enterprise_info |\n| **Tools** | analyze_trend, compare_enterprises |\n\n" +
    "**SQL 模板:**\n```sql\nSELECT year, quarter, SUM(revenue) as total_revenue,\n       SUM(profit) as total_profit,\n       ROUND(SUM(profit)/NULLIF(SUM(revenue),0)*100, 2) as profit_margin\nFROM financial_data WHERE enterprise_id = $1\nGROUP BY year, quarter ORDER BY year, quarter;\n```\n\n" +
    "### Skill 3: policy_match — 政策智能匹配\n" +
    "| 属性 | 说明 |\n|------|------|\n| **功能** | 根据企业特征智能匹配适用政策 |\n| **数据表** | policy_catalog JOIN enterprise_info |\n| **Tools** | match_policy, list_policies |\n\n" +
    "### Skill 5: risk_alert — 风险预警\n" +
    "| 风险类型 | 触发条件 | 等级 |\n|----------|----------|------|\n| 营收骤降 | 同比下降 > 30% | 高 |\n| 利润转负 | 连续 2 季度亏损 | 高 |\n| 税收异常 | 税收/营收比偏离行业均值 > 2σ | 中 |",
    { argsChunkSize: 80 },
  ));

  // 读 financial_data 表结构
  events.push(...toolCall(A, R, "run_shell_command",
    JSON.stringify({ command: "psql -h localhost -U nexau -d jxw_db -c \"\\\\d financial_data\"" }),
    "     Column     |  Type   | Nullable\n---------------+---------+----------\n id            | integer | not null\n enterprise_id | integer |\n year          | integer |\n quarter       | integer |\n revenue       | numeric |\n profit        | numeric |\n tax_amount    | numeric |",
  ));

  // 写 business_analysis tool
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "skills/business_analysis/analyze_trend.yaml",
      content: "name: analyze_trend\ndescription: 分析企业经营趋势（营收、利润）\nparameters:\n  type: object\n  required: [enterprise_id]\n  properties:\n    enterprise_id:\n      type: integer\n      description: 企业 ID\n    start_year:\n      type: integer\n      default: 2020\n    end_year:\n      type: integer\n      default: 2025\n",
    }),
    "File written: skills/business_analysis/analyze_trend.yaml",
    { argsChunkSize: 80 },
  ));

  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "skills/business_analysis/analyze_trend.py",
      content: "import asyncpg\n\nasync def analyze_trend(\n    enterprise_id: int,\n    start_year: int = 2020,\n    end_year: int = 2025,\n) -> dict:\n    conn = await asyncpg.connect(dsn=os.environ['DATABASE_URL'])\n    rows = await conn.fetch(\n        \"SELECT year, SUM(revenue) as revenue, SUM(profit) as profit \"\n        \"FROM financial_data WHERE enterprise_id=$1 AND year BETWEEN $2 AND $3 \"\n        \"GROUP BY year ORDER BY year\",\n        enterprise_id, start_year, end_year\n    )\n    await conn.close()\n    return {\n        'enterprise_id': enterprise_id,\n        'trend': [dict(r) for r in rows],\n    }\n",
    }),
    "File written: skills/business_analysis/analyze_trend.py",
    { argsChunkSize: 100 },
  ));

  // 更新 todos
  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "阅读 RFC 设计文档", status: "completed" },
        { description: "实现 business_analysis Skill", status: "completed" },
        { description: "实现 policy_match Skill", status: "in_progress" },
        { description: "编写测试用例", status: "pending" },
      ],
    }),
    "Todos updated",
    { extra: { taskSnapshot: IMPL_TASK_SNAPSHOTS[1] } },
  ));

  // 写 policy_match tool
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "skills/policy_match/match_policy.yaml",
      content: "name: match_policy\ndescription: 根据企业特征匹配适用政策\nparameters:\n  type: object\n  required: [enterprise_id]\n  properties:\n    enterprise_id:\n      type: integer\n      description: 企业 ID\n",
    }),
    "File written: skills/policy_match/match_policy.yaml",
    { argsChunkSize: 80 },
  ));

  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "skills/policy_match/match_policy.py",
      content: "import asyncpg\nfrom datetime import date\n\nasync def match_policy(enterprise_id: int) -> dict:\n    conn = await asyncpg.connect(dsn=os.environ['DATABASE_URL'])\n    enterprise = await conn.fetchrow(\n        \"SELECT * FROM enterprise_info WHERE id=$1\", enterprise_id\n    )\n    if not enterprise:\n        return {'error': 'Enterprise not found'}\n    policies = await conn.fetch(\n        \"SELECT * FROM policy_catalog WHERE \"\n        \"(target_industry IS NULL OR target_industry ILIKE $1) AND \"\n        \"(min_revenue IS NULL OR min_revenue <= $2) AND \"\n        \"(min_employees IS NULL OR min_employees <= $3) AND \"\n        \"effective_date <= $4 AND (expiry_date IS NULL OR expiry_date >= $4)\",\n        f\"%{enterprise['industry']}%\",\n        enterprise['registered_capital'],\n        enterprise['employee_count'],\n        date.today()\n    )\n    await conn.close()\n    return {\n        'enterprise': dict(enterprise),\n        'matched_policies': [dict(p) for p in policies],\n        'match_count': len(policies),\n    }\n",
    }),
    "File written: skills/policy_match/match_policy.py",
    { argsChunkSize: 100 },
  ));

  // 最终 todos
  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "阅读 RFC 设计文档", status: "completed" },
        { description: "实现 business_analysis Skill", status: "completed" },
        { description: "实现 policy_match Skill", status: "completed" },
        { description: "编写测试用例", status: "completed" },
      ],
    }),
    "Todos updated",
  ));

  events.push(...streamText(A, R,
    "两个 Skill 实现完成：\n" +
    "- **business_analysis**: `analyze_trend` tool — 营收/利润趋势查询\n" +
    "- **policy_match**: `match_policy` tool — 智能政策匹配\n\n" +
    "全部 Tool 定义和 Python 绑定已写入对应目录。",
    { startDelay: 200 },
  ));

  events.push(runFinished(A, R, 300));
  return events;
}

// ---------------------------------------------------------------------------
// Builder-Cfg: 配置 + 启动脚本
// ---------------------------------------------------------------------------
function buildBuilderCfgEvents(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];
  const A = BUILDER_CFG_ID;
  const R = BUILDER_CFG_ROLE;

  events.push(runStarted(A, R, 600));

  events.push(teamMessage(A, R, LEADER_ID,
    "CONFIG-001 + BUILD-001: 请编写智能体 nexau.yaml 配置和 run.py 启动脚本。", 200));

  events.push(...streamThinking(A, R,
    "我负责编写智能体配置文件和启动脚本。\n" +
    "需要将 4 个 Skills 集成到 nexau.yaml 中，\n" +
    "配置 LLM provider 和数据库连接。",
  ));

  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "编写 nexau.yaml 智能体配置", status: "in_progress" },
        { description: "编写 system prompt", status: "pending" },
        { description: "编写 run.py 启动脚本", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  // 读 RFC
  events.push(...toolCall(A, R, "read_file",
    JSON.stringify({ path: "rfcs/0001-jxw-agent.md" }),
    "# RFC-0001: 经信委数据分析智能体\n\n" +
    "## 智能体配置\n\n```yaml\nname: jxw-data-analyst\ndescription: 经信委数据分析智能体\n\nllm:\n  provider: openai\n  model: gpt-4o\n  temperature: 0.3\n\nskills:\n  - enterprise_query\n  - business_analysis\n  - policy_match\n  - supply_chain\n  - risk_alert\n\nenv:\n  DATABASE_URL: ${DATABASE_URL}\n```\n\n" +
    "## 实现计划\n\n| 阶段 | 任务 | 负责 |\n|------|------|------|\n| Phase 1 | 数据库探索 + enterprise_query Skill | Builder-DB |\n| Phase 2 | business_analysis + policy_match Skill | Builder-Skill |\n| Phase 3 | supply_chain + risk_alert Skill | Builder-Skill |\n| Phase 4 | 智能体配置 + 启动脚本 | Builder-Cfg |\n| Phase 5 | 集成测试 | Builder-DB |",
    { argsChunkSize: 80 },
  ));

  // 写 nexau.yaml
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "nexau.yaml",
      content: "name: jxw-data-analyst\ndescription: 经信委数据分析智能体\n\nllm:\n  provider: openai\n  model: gpt-4o\n  temperature: 0.3\n\nsystem_prompt: systemprompt.md\n\nskills:\n  - enterprise_query\n  - business_analysis\n  - policy_match\n\ntools:\n  - name: search_enterprise\n    skill: enterprise_query\n  - name: analyze_trend\n    skill: business_analysis\n  - name: match_policy\n    skill: policy_match\n\nenv:\n  DATABASE_URL: ${DATABASE_URL}\n",
    }),
    "File written: nexau.yaml",
    { argsChunkSize: 80 },
  ));

  // 写 system prompt
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "systemprompt.md",
      content: "# 经信委数据分析智能体\n\n你是一个专业的经信委数据分析助手。你可以访问经信委的 PostgreSQL 数据库，帮助用户进行企业信息查询、经营数据分析和政策匹配。\n\n## 你的能力\n\n1. **企业查询** — 使用 search_enterprise 工具按名称、行业、地区检索企业\n2. **经营分析** — 使用 analyze_trend 工具分析企业营收和利润趋势\n3. **政策匹配** — 使用 match_policy 工具为企业匹配适用政策\n\n## 工作原则\n\n- 始终先确认用户的查询意图\n- 展示数据时使用表格格式\n- 分析结论要有数据支撑\n- 政策匹配要说明匹配条件\n",
    }),
    "File written: systemprompt.md",
    { argsChunkSize: 100 },
  ));

  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "编写 nexau.yaml 智能体配置", status: "completed" },
        { description: "编写 system prompt", status: "completed" },
        { description: "编写 run.py 启动脚本", status: "in_progress" },
      ],
    }),
    "Todos updated",
  ));

  // 写 run.py
  events.push(...toolCall(A, R, "write_file",
    JSON.stringify({
      file_path: "run.py",
      content: "#!/usr/bin/env python3\n\"\"\"经信委数据分析智能体 — 启动入口\"\"\"\nimport asyncio\nfrom nexau import Agent, SSETransportServer\n\nasync def main():\n    agent = Agent.from_config('nexau.yaml')\n    server = SSETransportServer(agent, host='0.0.0.0', port=8000)\n    print('经信委数据分析智能体已启动: http://localhost:8000')\n    await server.serve()\n\nif __name__ == '__main__':\n    asyncio.run(main())\n",
    }),
    "File written: run.py",
    { argsChunkSize: 80 },
  ));

  // 最终 todos
  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "编写 nexau.yaml 智能体配置", status: "completed" },
        { description: "编写 system prompt", status: "completed" },
        { description: "编写 run.py 启动脚本", status: "completed" },
      ],
    }),
    "Todos updated",
    { extra: { taskSnapshot: IMPL_TASK_SNAPSHOTS[2] } },
  ));

  events.push(...streamText(A, R,
    "配置和启动脚本已完成：\n" +
    "- `nexau.yaml` — 智能体配置（集成 3 个 Skills）\n" +
    "- `systemprompt.md` — 系统提示词\n" +
    "- `run.py` — 启动入口脚本",
    { startDelay: 200 },
  ));

  events.push(runFinished(A, R, 300));
  return events;
}

// ---------------------------------------------------------------------------
// Testing: Builder-DB 运行集成测试
// ---------------------------------------------------------------------------
function buildTestingEvents(): TimedEnvelope[] {
  const events: TimedEnvelope[] = [];
  const A = BUILDER_DB_ID;
  const R = BUILDER_DB_ROLE;

  // Builder-DB 被重新激活来执行测试
  events.push(runStarted(A, R, 800));

  events.push(teamMessage(A, R, LEADER_ID,
    "TEST-001: 请运行集成测试验证所有 Skills 正常工作。", 200));

  events.push(...streamThinking(A, R,
    "需要运行集成测试验证所有 Skills：\n" +
    "1. enterprise_query — 搜索测试\n" +
    "2. business_analysis — 趋势分析测试\n" +
    "3. policy_match — 政策匹配测试\n" +
    "4. 启动脚本加载验证",
  ));

  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "验证 enterprise_query", status: "in_progress" },
        { description: "验证 business_analysis", status: "pending" },
        { description: "验证 policy_match", status: "pending" },
        { description: "验证启动脚本", status: "pending" },
      ],
    }),
    "Todos updated",
  ));

  events.push(...toolCall(A, R, "run_shell_command",
    JSON.stringify({ command: "python -m pytest tests/ -v --tb=short" }),
    "tests/test_enterprise_query.py::test_search_by_name PASSED\n" +
    "tests/test_enterprise_query.py::test_search_by_industry PASSED\n" +
    "tests/test_enterprise_query.py::test_search_by_region PASSED\n" +
    "tests/test_business_analysis.py::test_trend_analysis PASSED\n" +
    "tests/test_business_analysis.py::test_empty_data PASSED\n" +
    "tests/test_policy_match.py::test_match_found PASSED\n" +
    "tests/test_policy_match.py::test_no_match PASSED\n\n" +
    "===================== 7 passed in 3.21s =====================",
    { startDelay: 400 },
  ));

  events.push(...toolCall(A, R, "run_shell_command",
    JSON.stringify({ command: "python -c \"from nexau import Agent; a = Agent.from_config('nexau.yaml'); print('Config loaded:', a.name)\"" }),
    "Config loaded: jxw-data-analyst",
  ));

  events.push(...toolCall(A, R, "write_todos",
    JSON.stringify({
      todos: [
        { description: "验证 enterprise_query", status: "completed" },
        { description: "验证 business_analysis", status: "completed" },
        { description: "验证 policy_match", status: "completed" },
        { description: "验证启动脚本", status: "completed" },
      ],
    }),
    "Todos updated",
  ));

  events.push(...streamText(A, R,
    "集成测试全部通过 ✓\n\n" +
    "- 7 项测试用例全部 PASSED\n" +
    "- 智能体配置加载验证成功\n" +
    "- 所有 Skills 功能正常",
    { startDelay: 200 },
  ));

  events.push(runFinished(A, R, 300));
  return events;
}

// ═══════════════════════════════════════════════════════════════════════════
//  导出: 按 Phase 获取事件
// ═══════════════════════════════════════════════════════════════════════════
const PHASE_EVENTS_CACHE: Partial<Record<Phase, TimedEnvelope[]>> = {};

export function getPhaseEvents(phase: Phase): TimedEnvelope[] {
  if (PHASE_EVENTS_CACHE[phase]) return PHASE_EVENTS_CACHE[phase]!;

  let events: TimedEnvelope[];
  switch (phase) {
    case Phase.REQUIREMENTS:
      events = buildRequirementsPhase();
      break;
    case Phase.DESIGN:
      events = buildDesignPhase();
      break;
    case Phase.IMPLEMENTATION:
      events = buildImplementationPhase();
      break;
    default:
      events = [];
  }

  PHASE_EVENTS_CACHE[phase] = events;
  return events;
}
