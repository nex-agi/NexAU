"""RFC-0024 plugin adapter.

The adapter is a load-time bridge: it expands ``plugins`` entries from an
agent YAML into existing AgentConfig fields. It does not introduce runtime
plugin primitives.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import ValidationError

from nexau.archs.main_sub.config.expanded import ExpandedSubAgentConfig
from nexau.archs.main_sub.utils import load_yaml_text_with_vars

from .manifest import PluginConfigProperty, PluginManifest

logger = logging.getLogger(__name__)


class PluginConfigError(Exception):
    """Raised when a plugin manifest or expansion violates RFC-0024."""


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")
_CONFIG_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SOURCE_ID_LIMIT = 256
_DISALLOWED_CONTRIBUTIONS = {"tracers", "token_counter", "before_model_hooks", "after_model_hooks", "before_tool_hooks", "after_tool_hooks"}


@dataclass(frozen=True)
class ExpandedPluginContributions:
    """Plugin contributions after variable rendering and path resolution."""

    system_prompt_fragments: list[str]
    mcp_servers: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    sub_agents: list[ExpandedSubAgentConfig]
    middlewares: list[dict[str, Any]]


class PluginAdapter:
    """Expand top-level agent ``plugins`` into normal agent config fields."""

    def __init__(self, config: dict[str, Any], *, agent_base_path: Path, strict: bool = True):
        self._config: dict[str, Any] = deepcopy(config)
        self._agent_base_path = agent_base_path
        self._strict = strict
        self._skipped_components: list[str] = []
        self._seen_names: dict[tuple[str, str], str] = {}

    def expand(self) -> dict[str, Any]:
        """Return config with plugin contributions merged and ``plugins`` removed."""
        plugin_entries_raw: object = self._config.get("plugins") or []
        if not isinstance(plugin_entries_raw, list):
            raise PluginConfigError("'plugins' must be a list")
        plugin_entries = cast(list[object], plugin_entries_raw)

        self._register_local_sources()

        seen_uses: set[str] = set()
        plugin_mcp_servers: list[dict[str, Any]] = []
        plugin_tools: list[dict[str, Any]] = []
        plugin_skills: list[dict[str, Any]] = []
        plugin_sub_agents: list[ExpandedSubAgentConfig] = []
        plugin_middlewares: list[dict[str, Any]] = []
        plugin_system_prompt_fragments: list[str] = []

        for index, entry_obj in enumerate(plugin_entries):
            if not isinstance(entry_obj, dict):
                raise PluginConfigError(f"plugins[{index}] must be a mapping")
            entry = cast(dict[str, Any], entry_obj)
            use_value = entry.get("use")
            if not isinstance(use_value, str) or not use_value:
                raise PluginConfigError(f"plugins[{index}].use is required")
            if use_value in seen_uses:
                raise PluginConfigError(f"Duplicate plugin use '{use_value}' in the same agent YAML")
            seen_uses.add(use_value)

            plugin_dir = self._resolve_plugin_use(use_value)
            manifest = self._load_manifest(plugin_dir)
            provided_config_raw: object = entry.get("config") or {}
            if not isinstance(provided_config_raw, dict):
                raise PluginConfigError(f"plugins[{index}].config must be a mapping")
            resolved_config = self._resolve_config(manifest, cast(dict[str, Any], provided_config_raw))

            expanded = self._expand_manifest(
                manifest=manifest,
                plugin_dir=plugin_dir,
                resolved_config=resolved_config,
            )
            plugin_mcp_servers.extend(expanded.mcp_servers)
            plugin_tools.extend(expanded.tools)
            plugin_skills.extend(expanded.skills)
            plugin_sub_agents.extend(expanded.sub_agents)
            plugin_middlewares.extend(expanded.middlewares)
            plugin_system_prompt_fragments.extend(expanded.system_prompt_fragments)

        expanded_config = deepcopy(self._config)
        expanded_config.pop("plugins", None)
        expanded_config["mcp_servers"] = list(expanded_config.get("mcp_servers") or []) + plugin_mcp_servers
        expanded_config["tools"] = list(expanded_config.get("tools") or []) + plugin_tools
        expanded_config["skills"] = list(expanded_config.get("skills") or []) + plugin_skills
        expanded_config["sub_agents"] = list(expanded_config.get("sub_agents") or []) + plugin_sub_agents
        expanded_config["middlewares"] = plugin_middlewares + list(expanded_config.get("middlewares") or [])
        combined_suffix = self._combine_system_prompt_suffix(
            plugin_system_prompt_fragments,
            expanded_config.get("system_prompt_suffix"),
        )
        if combined_suffix is not None:
            expanded_config["system_prompt_suffix"] = combined_suffix
        if self._skipped_components:
            expanded_config["_skipped_components"] = list(expanded_config.get("_skipped_components") or []) + self._skipped_components
        return expanded_config

    def _register_local_sources(self) -> None:
        tools_raw: object = self._config.get("tools") or []
        if isinstance(tools_raw, list):
            for tool_obj in cast(list[object], tools_raw):
                if isinstance(tool_obj, dict):
                    tool = cast(dict[str, Any], tool_obj)
                    name = tool.get("name")
                    if isinstance(name, str):
                        source_id = tool.get("source_id")
                        if not isinstance(source_id, str):
                            source_id = self._source_id("local", "tool", name)
                            tool["source_id"] = source_id
                        self._claim_name("tool", name, source_id)

        servers_raw: object = self._config.get("mcp_servers") or []
        if isinstance(servers_raw, list):
            for server_obj in cast(list[object], servers_raw):
                if isinstance(server_obj, dict):
                    server = cast(dict[str, Any], server_obj)
                    name = server.get("name")
                    if isinstance(name, str):
                        source_id = server.get("source_id")
                        if not isinstance(source_id, str):
                            source_id = self._source_id("local", "mcp_server", name)
                            server["source_id"] = source_id
                        self._claim_name("mcp_server", name, source_id)

        sub_agents_raw: object = self._config.get("sub_agents") or []
        if isinstance(sub_agents_raw, list):
            for sub_agent_obj in cast(list[object], sub_agents_raw):
                if isinstance(sub_agent_obj, dict):
                    sub_agent = cast(dict[str, Any], sub_agent_obj)
                    name = sub_agent.get("name")
                    if isinstance(name, str):
                        source_id = sub_agent.get("source_id")
                        if not isinstance(source_id, str):
                            source_id = self._source_id("local", "sub_agent", name)
                            sub_agent["source_id"] = source_id
                        self._claim_name("sub_agent", name, source_id)

    def _resolve_plugin_use(self, use_value: str) -> Path:
        if ":" not in use_value:
            raise PluginConfigError("plugins[].use must use '<scheme>:<body>' format; Phase 1 supports only 'path:'")
        scheme, body = use_value.split(":", 1)
        if scheme != "path":
            raise PluginConfigError(f"Plugin URI scheme '{scheme}' is not supported in Phase 1")
        if not body:
            raise PluginConfigError("path: plugin URI must include a path body")
        path = Path(body)
        plugin_dir = path if path.is_absolute() else self._agent_base_path / path
        return plugin_dir.resolve(strict=False)

    def _load_manifest(self, plugin_dir: Path) -> PluginManifest:
        manifest_path = plugin_dir / "plugin.yaml"
        if not manifest_path.exists():
            raise PluginConfigError(f"plugin.yaml not found: {manifest_path}")
        try:
            raw_obj: object = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw_obj, dict):
                raise PluginConfigError(f"plugin.yaml must be a mapping: {manifest_path}")
            raw = cast(dict[str, Any], raw_obj)
            contributes_raw: object = raw.get("contributes") or {}
            contributes = cast(dict[str, Any], contributes_raw) if isinstance(contributes_raw, dict) else {}
            disallowed = _DISALLOWED_CONTRIBUTIONS & set(contributes)
            if disallowed:
                raise PluginConfigError(f"Plugin contributes unsupported fields: {sorted(disallowed)}")
            manifest = PluginManifest.model_validate(raw)
        except ValidationError as exc:
            raise PluginConfigError(f"Invalid plugin manifest {manifest_path}: {exc}") from exc
        self._validate_engine(manifest)
        return manifest

    def _validate_engine(self, manifest: PluginManifest) -> None:
        current = _current_nexau_version()
        spec = manifest.engines.nexau.strip()
        if not spec:
            raise PluginConfigError(f"Plugin '{manifest.name}' engines.nexau must not be empty")
        try:
            current_version = Version(current)
            specifier = SpecifierSet(spec)
        except (InvalidVersion, InvalidSpecifier) as exc:
            raise PluginConfigError(f"Plugin '{manifest.name}' has invalid NexAU engine spec '{manifest.engines.nexau}': {exc}") from exc
        if current_version not in specifier:
            raise PluginConfigError(
                f"Plugin '{manifest.name}' requires NexAU '{manifest.engines.nexau}', current version is {current}",
            )

    def _resolve_config(self, manifest: PluginManifest, provided: dict[str, Any]) -> dict[str, Any]:
        properties = manifest.config.properties
        invalid_names = [name for name in properties if not _CONFIG_NAME_PATTERN.fullmatch(name)]
        if invalid_names:
            raise PluginConfigError(
                f"Plugin '{manifest.name}' config property names must match {_CONFIG_NAME_PATTERN.pattern}: {sorted(invalid_names)}",
            )
        unknown = set(provided) - set(properties)
        if unknown:
            raise PluginConfigError(f"Plugin '{manifest.name}' received unknown config keys: {sorted(unknown)}")

        resolved: dict[str, Any] = {}
        for name, prop in properties.items():
            if name in provided:
                value = provided[name]
            elif prop.required:
                raise PluginConfigError(f"Plugin '{manifest.name}' missing required config '{name}'")
            elif prop.default is not None:
                value = prop.default
            else:
                continue
            self._validate_config_value(manifest.name, name, prop, value)
            resolved[name] = value
        return resolved

    def _validate_config_value(self, plugin_name: str, name: str, prop: PluginConfigProperty, value: Any) -> None:
        type_name = prop.type
        ok = (
            (type_name == "string" and isinstance(value, str))
            or (type_name == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (type_name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (type_name == "boolean" and isinstance(value, bool))
            or (
                type_name == "string_array" and isinstance(value, list) and all(isinstance(item, str) for item in cast(list[object], value))
            )
        )
        if not ok:
            raise PluginConfigError(f"Plugin '{plugin_name}' config '{name}' must be {type_name}")
        if prop.enum is not None and value not in prop.enum:
            raise PluginConfigError(f"Plugin '{plugin_name}' config '{name}' must be one of {prop.enum}")

    def _expand_manifest(
        self,
        *,
        manifest: PluginManifest,
        plugin_dir: Path,
        resolved_config: dict[str, Any],
    ) -> ExpandedPluginContributions:
        context = {"plugin.dir": str(plugin_dir), **{f"config.{key}": value for key, value in resolved_config.items()}}
        return ExpandedPluginContributions(
            system_prompt_fragments=self._expand_system_prompt_fragment(manifest, context),
            mcp_servers=self._expand_mcp_servers(manifest, plugin_dir, context),
            tools=self._expand_tools(manifest, plugin_dir, context),
            skills=self._expand_skills(manifest, plugin_dir, context),
            sub_agents=self._expand_sub_agents(manifest, plugin_dir, context),
            middlewares=self._expand_middlewares(manifest, plugin_dir, context),
        )

    def _expand_system_prompt_fragment(self, manifest: PluginManifest, context: dict[str, Any]) -> list[str]:
        raw_fragment = manifest.system_prompt_fragment
        if raw_fragment is None:
            return []

        rendered_fragment = self._render_value(raw_fragment, context)
        if not isinstance(rendered_fragment, str):
            raise PluginConfigError(f"Plugin '{manifest.name}' system_prompt_fragment must render to a string")

        fragment = rendered_fragment.strip()
        if not fragment:
            return []
        return [fragment]

    def _expand_mcp_servers(self, manifest: PluginManifest, plugin_dir: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for server_model in manifest.contributes.mcp_servers:
            raw_server = server_model.model_dump(mode="python", by_alias=True, exclude_none=True)
            # RFC-0029: plugins may declare an environment-backed secret
            # reference, but ordinary plugin config interpolation must never
            # become an OAuth token or client secret.
            raw_auth = raw_server.get("auth")
            raw_auth_dict = cast(dict[str, Any], raw_auth) if isinstance(raw_auth, dict) else None
            secret_values = (
                (
                    raw_auth_dict.get("token"),
                    raw_auth_dict.get("client_secret"),
                )
                if raw_auth_dict is not None
                else ()
            )
            if any("${config." in repr(value) for value in secret_values):
                raise PluginConfigError(
                    f"Plugin '{manifest.name}' MCP auth secrets must use a static secret reference; config interpolation is not allowed",
                )
            server = self._render_value(raw_server, context)
            name = server["name"]
            source_id = self._source_id(f"plugin:{manifest.name}", "mcp_server", name)
            self._claim_name("mcp_server", name, source_id)
            if Path(str(server.get("command", ""))).is_absolute():
                raise PluginConfigError(f"Plugin '{manifest.name}' mcp server '{name}' command must not be absolute")
            server["source_id"] = source_id
            expanded.append(server)
        return expanded

    def _expand_tools(self, manifest: PluginManifest, plugin_dir: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for tool_model in manifest.contributes.tools:
            tool = self._render_value(tool_model.model_dump(mode="python", exclude_none=True), context)
            name = tool["name"]
            source_id = self._source_id(f"plugin:{manifest.name}", "tool", name)
            self._claim_name("tool", name, source_id)
            yaml_path = self._resolve_inside_plugin(plugin_dir, str(tool["yaml_path"]), field=f"tool '{name}' yaml_path")
            tool["yaml_path"] = str(yaml_path)
            binding = tool.get("binding")
            if isinstance(binding, str):
                tool["binding"] = self._resolve_import_string(plugin_dir, binding)
            tool["source_id"] = source_id
            self._check_tool_extra_kwargs(tool, yaml_path, source_id)
            expanded.append(tool)
        return expanded

    def _expand_skills(self, manifest: PluginManifest, plugin_dir: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for skill_model in manifest.contributes.skills:
            skill = self._render_value(skill_model.model_dump(mode="python"), context)
            name = skill["name"]
            source_id = self._source_id(f"plugin:{manifest.name}", "skill", name)
            self._claim_name("skill", name, source_id)
            skill["path"] = str(self._resolve_inside_plugin(plugin_dir, str(skill["path"]), field=f"skill '{name}' path"))
            skill["source_id"] = source_id
            expanded.append(skill)
        return expanded

    def _expand_sub_agents(self, manifest: PluginManifest, plugin_dir: Path, context: dict[str, Any]) -> list[ExpandedSubAgentConfig]:
        expanded: list[ExpandedSubAgentConfig] = []
        for sub_model in manifest.contributes.sub_agents:
            sub_agent = self._render_value(sub_model.model_dump(mode="python", exclude_none=True), context)
            name = sub_agent["name"]
            source_id = self._source_id(f"plugin:{manifest.name}", "sub_agent", name)
            self._claim_name("sub_agent", name, source_id)
            config_path = self._resolve_inside_plugin(plugin_dir, str(sub_agent["config_path"]), field=f"sub-agent '{name}' config_path")
            try:
                # Sub-agent YAML should remain a reusable agent config, so only
                # plugin parameters are injected and the plugin install path is not.
                sub_agent_context = {key: value for key, value in context.items() if key != "plugin.dir"}
                rendered_text = self._render_string(config_path.read_text(encoding="utf-8"), sub_agent_context)
                loaded = load_yaml_text_with_vars(rendered_text, config_path.parent)
                if not isinstance(loaded, dict):
                    raise PluginConfigError(f"Sub-agent '{name}' config must be a mapping")
            except Exception as exc:
                self._component_error(f"Skipped sub-agent '{name}' from plugin '{manifest.name}': {exc}", exc)
                continue
            expanded.append(
                ExpandedSubAgentConfig(
                    name=str(name),
                    config_path=str(config_path),
                    source_id=source_id,
                    inline_config=loaded,
                    inline_base_path=config_path.parent,
                ),
            )
        return expanded

    def _expand_middlewares(self, manifest: PluginManifest, plugin_dir: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for middleware_model in manifest.contributes.middlewares:
            middleware = self._render_value(middleware_model.model_dump(mode="python", by_alias=True, exclude_none=True), context)
            name = middleware["name"]
            source_id = self._source_id(f"plugin:{manifest.name}", "middleware", name)
            self._claim_name("middleware", name, source_id)
            middleware["import"] = self._resolve_import_string(plugin_dir, str(middleware["import"]))
            middleware["source_id"] = source_id
            expanded.append(middleware)
        return expanded

    def _render_value(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, str):
            return self._render_string(value, context)
        if isinstance(value, list):
            return [self._render_value(item, context) for item in cast(list[Any], value)]
        if isinstance(value, dict):
            return {key: self._render_value(item, context) for key, item in cast(dict[str, Any], value).items()}
        return value

    def _render_string(self, value: str, context: dict[str, Any]) -> Any:
        matches = list(_VAR_PATTERN.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            key = matches[0].group(1)
            if key not in context:
                raise PluginConfigError(f"Unsupported or undefined plugin variable '${{{key}}}'")
            return context[key]

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in context:
                raise PluginConfigError(f"Unsupported or undefined plugin variable '${{{key}}}'")
            return str(context[key])

        return _VAR_PATTERN.sub(replace, value)

    def _combine_system_prompt_suffix(self, plugin_fragments: list[str], local_suffix: object) -> str | None:
        parts = [fragment.strip() for fragment in plugin_fragments if fragment.strip()]
        if not parts:
            return local_suffix if isinstance(local_suffix, str) else None
        if isinstance(local_suffix, str) and local_suffix.strip():
            parts.append(local_suffix.strip())
        return "\n\n".join(parts)

    def _resolve_inside_plugin(self, plugin_dir: Path, raw_path: str, *, field: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            candidate = path.resolve(strict=False)
        else:
            candidate = (plugin_dir / path).resolve(strict=False)
        plugin_root = plugin_dir.resolve(strict=False)
        try:
            candidate.relative_to(plugin_root)
        except ValueError as exc:
            raise PluginConfigError(f"Plugin path for {field} escapes plugin root: {raw_path}") from exc
        return candidate

    def _resolve_import_string(self, plugin_dir: Path, import_string: str) -> str:
        if ":" not in import_string:
            raise PluginConfigError(f"Import string must contain ':' separator: {import_string}")
        module_part, attr = import_string.rsplit(":", 1)
        if module_part.endswith(".py") or module_part.startswith(".") or "/" in module_part or "\\" in module_part:
            module_path = self._resolve_inside_plugin(plugin_dir, module_part, field="import")
            return f"{module_path}:{attr}"
        candidate = plugin_dir / (module_part.replace(".", "/") + ".py")
        if candidate.exists():
            return f"{candidate.resolve(strict=False)}:{attr}"
        return import_string

    def _check_tool_extra_kwargs(self, tool: dict[str, Any], yaml_path: Path, source_id: str) -> None:
        extra_kwargs_raw: object = tool.get("extra_kwargs") or {}
        extra_kwargs = cast(dict[str, Any], extra_kwargs_raw) if isinstance(extra_kwargs_raw, dict) else {}
        if not yaml_path.exists():
            return
        try:
            raw_obj: object = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            raw = cast(dict[str, Any], raw_obj) if isinstance(raw_obj, dict) else {}
            input_schema_raw: object = raw.get("input_schema")
            input_schema = cast(dict[str, Any], input_schema_raw) if isinstance(input_schema_raw, dict) else {}
            properties_raw: object = input_schema.get("properties", {})
            properties = cast(dict[str, Any], properties_raw) if isinstance(properties_raw, dict) else {}
            conflicts = set(extra_kwargs) & set(properties)
            if conflicts:
                raise PluginConfigError(f"Tool '{tool['name']}' extra_kwargs conflicts with input_schema keys: {sorted(conflicts)}")
        except PluginConfigError:
            raise
        except Exception as exc:
            self._component_error(f"Skipped tool schema precheck for {source_id}: {exc}", exc)

    def _claim_name(self, kind: str, name: str, source_id: str) -> None:
        key = (kind, name)
        previous = self._seen_names.get(key)
        if previous is not None:
            raise PluginConfigError(
                f"{kind} name conflict for '{name}' between {previous} and {source_id}. "
                "Rename the manifest/local resource or avoid enabling both plugins.",
            )
        self._seen_names[key] = source_id

    def _source_id(self, source_prefix: str, kind: str, resource_name: str) -> str:
        if source_prefix == "local":
            source_id = f"local:{kind}:{resource_name}"
        else:
            source_id = f"{source_prefix}:{kind}:{resource_name}"
        if len(source_id) > _SOURCE_ID_LIMIT:
            raise PluginConfigError(f"source_id exceeds {_SOURCE_ID_LIMIT} characters: {source_id}")
        return source_id

    def _component_error(self, message: str, error: Exception) -> None:
        if self._strict:
            raise PluginConfigError(message) from error
        logger.warning(message, extra={"source_id": None})
        self._skipped_components.append(message)


def _current_nexau_version() -> str:
    try:
        return version("nexau")
    except PackageNotFoundError as exc:
        raise PluginConfigError("Unable to determine installed NexAU version from package metadata") from exc
