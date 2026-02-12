"""Tests for the plugin registry."""

from __future__ import annotations

import pytest

from awe_agent.plugins.registry import Registry


class DummyBase:
    pass


class DummyA(DummyBase):
    pass


class DummyB(DummyBase):
    pass


def test_register_and_get():
    reg: Registry[DummyBase] = Registry("test.group")
    reg.register("a", DummyA)
    assert reg.get("a") is DummyA


def test_register_decorator():
    reg: Registry[DummyBase] = Registry("test.group")

    @reg.decorator("b")
    class B(DummyBase):
        pass

    assert reg.get("b") is B


def test_get_missing_raises():
    reg: Registry[DummyBase] = Registry("test.group.nonexistent")
    with pytest.raises(KeyError, match="not_registered"):
        reg.get("not_registered")


def test_register_overwrite():
    reg: Registry[DummyBase] = Registry("test.group")
    reg.register("x", DummyA)
    reg.register("x", DummyB)
    assert reg.get("x") is DummyB


def test_list_available():
    reg: Registry[DummyBase] = Registry("test.group")
    reg.register("first", DummyA)
    reg.register("second", DummyB)
    names = reg.list_available()
    assert "first" in names
    assert "second" in names


def test_entry_points_discovery():
    """Verify that built-in entry points are discoverable."""
    reg: Registry[type] = Registry("awe_agent.runtime")
    # The docker runtime should be registered via entry_points
    docker_cls = reg.get("docker")
    assert docker_cls is not None
    assert docker_cls.__name__ == "DockerRuntime"
