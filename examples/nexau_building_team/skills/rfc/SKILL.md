---
name: rfc-writer
description: Guide for writing RFC (Request for Comments) documents following NexAU project conventions. This skill should be used when users want to create a new RFC, update an existing RFC, or need guidance on RFC structure, formatting, and best practices.
---

# RFC Writer

This skill provides guidance for creating well-structured RFC documents that follow NexAU project conventions.

## When to Use

- Creating a new RFC for a feature, architecture change, or technical decision
- Updating an existing RFC (status change, content revision)
- Reviewing RFC structure and completeness

## RFC Creation Workflow

### 1. Determine the RFC Number

Read `README.md` in the RFC directory to find the current RFC list. Use the next available 4-digit number (e.g., if the last RFC is `0002`, use `0003`).

### 2. Create the RFC File

Copy `0000-template.md` as the starting point. Name the file `{number}-{short-title}.md` using lowercase English words separated by hyphens.

Example: `0003-session-recovery.md`

### 3. Fill in the Content

Refer to `WRITING_GUIDE.md` for detailed formatting rules. Key requirements:

**Title**: `# RFC-{number}: {中文标题}` — use Chinese for readability.

**Front Matter** (all fields required):
- **状态**: draft | accepted | implemented | superseded | rejected
- **优先级**: P0 | P1 | P2 | P3
- **标签**: `agent`, `tool`, `skill`, 等
- **Agent 角色**: Agent 角色名如 `multi-modal-extractor`, `ads-judger` 等
- **创建日期**: YYYY-MM-DD
- **更新日期**: YYYY-MM-DD

**Required Sections**: 摘要, 动机, 设计 (概述 + 详细设计), 权衡取舍 (替代方案 + 缺点), 实现计划, 未解决的问题

**Optional Sections**: 示例, 测试方案, 相关文件, 参考资料

### 4. Content Depth

RFC is a design document, not an implementation document. Follow these principles:

- Include: architecture diagrams, API interface definitions, data models, key algorithm ideas, tech choices with rationale
- Exclude: full code implementations, internal function details, ORM mapping code, complete config files, test code

Code examples should be kept to 3-5 lines demonstrating API usage, config format, or data structure definitions.

### 5. Diagrams

Use Mermaid for diagrams when possible, following the color conventions:

- 🟢 Completed/Trusted: `#10B981` / `#059669`
- 🟠 In Progress: `#F59E0B` / `#D97706`
- 🔵 Test Code: `#3B82F6` / `#2563EB`
- 🔴 Error/Untrusted: `#EF4444` / `#DC2626`
- 🟣 Docker/Container: `#8B5CF6` / `#7C3AED`
- 🔷 Gateway/Protocol: `#06B6D4` / `#0891B2`
- 🩵 Storage/Database: `#14B8A6` / `#0D9488`

Fall back to ASCII diagrams when Mermaid is not suitable.

### 6. Update the Index

After creating the RFC, update `README.md` to add the new RFC entry to the appropriate category table.

### 7. Pre-Submission Checklist

Before finalizing, verify:

- File name follows `{number}-{title}.md` format
- Title follows `# RFC-{number}: {中文标题}` format
- All front matter fields are present
- All required sections are included
- Diagrams use the standard color scheme
- Code examples are concise (3-5 lines)
- `README.md` index is updated

## Bundled Resources

| File | Purpose |
|------|---------|
| `0000-template.md` | RFC template — copy this to start a new RFC |
| `README.md` | RFC index — check for next available number, update after creating |
| `WRITING_GUIDE.md` | Full formatting specification — consult for detailed rules on structure, diagrams, tables, and code examples |
