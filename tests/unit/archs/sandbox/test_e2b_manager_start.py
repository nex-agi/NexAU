from types import SimpleNamespace

import pytest

from nexau.archs.sandbox.base_sandbox import E2BSandboxConfig
from nexau.archs.sandbox.e2b_sandbox import E2BSandbox, E2BSandboxManager


@pytest.fixture
def isolated_create(monkeypatch):
    """Replace remote E2B calls while retaining the manager's start control flow."""

    import nexau.archs.sandbox.e2b_sandbox as module

    raw = SimpleNamespace(sandbox_id="sandbox-test", _envd_version="0.1.4")

    class FakeSandboxApi:
        @staticmethod
        def beta_create(**_kwargs):
            return raw

    created_directories: list[str] = []
    monkeypatch.setattr(module, "E2B_AVAILABLE", True)
    monkeypatch.setattr(module, "Sandbox", FakeSandboxApi)
    monkeypatch.setattr(E2BSandboxManager, "load_sandbox_state", lambda *_args: None)
    monkeypatch.setattr(E2BSandboxManager, "persist_sandbox_state", lambda *_args: None)
    monkeypatch.setattr(E2BSandboxManager, "_start_keepalive", lambda *_args: None)
    monkeypatch.setattr(E2BSandboxManager, "_maybe_rebuild_for_http", lambda _self, value, _config: value)
    monkeypatch.setattr(
        E2BSandbox,
        "create_directory",
        lambda _self, path, **_kwargs: created_directories.append(path) or True,
    )
    return created_directories


def test_self_host_default_work_dir_does_not_issue_redundant_remote_mkdir(isolated_create):
    manager = E2BSandboxManager()

    sandbox = manager.start(
        session_manager=None,
        user_id="user",
        session_id="session",
        sandbox_config=E2BSandboxConfig(work_dir="/home/user", force_http=True, keepalive_interval=0),
    )

    assert sandbox.sandbox_id == "sandbox-test"
    assert isolated_create == []


def test_self_host_custom_work_dir_is_still_created_remotely(isolated_create):
    manager = E2BSandboxManager()

    sandbox = manager.start(
        session_manager=None,
        user_id="user",
        session_id="session",
        sandbox_config=E2BSandboxConfig(work_dir="/home/user/custom", force_http=True, keepalive_interval=0),
    )

    assert sandbox.sandbox_id == "sandbox-test"
    assert isolated_create == ["/home/user/custom"]


def test_saas_default_work_dir_keeps_remote_ensure(isolated_create):
    manager = E2BSandboxManager()

    manager.start(
        session_manager=None,
        user_id="user",
        session_id="session",
        sandbox_config=E2BSandboxConfig(work_dir="/home/user", force_http=False, keepalive_interval=0),
    )

    assert isolated_create == ["/home/user"]
