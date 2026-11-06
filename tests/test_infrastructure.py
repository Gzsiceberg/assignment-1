"""Fast checks for setup helpers that must work before student TODOs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from assignment.env import Environment, _tls_port_configuration


def test_runtime_port_uses_tls_and_is_not_duplicated():
    encrypted, unencrypted, extra = _tls_port_configuration(
        {
            "encrypted_ports": [8000, 9000],
            "unencrypted_ports": [7000, 8000],
            "cpu": 2,
        },
        8000,
    )

    assert encrypted == [8000, 9000]
    assert unencrypted == [7000]
    assert extra == {"cpu": 2}


def test_runtime_port_is_added_when_no_port_lists_are_given():
    encrypted, unencrypted, extra = _tls_port_configuration({}, 8000)

    assert encrypted == [8000]
    assert unencrypted == []
    assert extra == {}


def test_default_environment_uses_modal_image(monkeypatch):
    image = Mock()
    debian_slim = Mock(return_value=image)
    monkeypatch.setattr("assignment.env.modal.Image.debian_slim", debian_slim)
    deployment = SimpleNamespace(start=AsyncMock(), is_alive=AsyncMock())
    constructor = Mock(return_value=deployment)
    monkeypatch.setattr("assignment.env.AssignmentModalDeployment", constructor)
    monkeypatch.setattr(Environment, "execute", lambda *a, **kw: {"output": "Linux\nr\nv\nx86_64"})

    Environment()

    debian_slim.assert_called_once_with(python_version="3.12")
    assert constructor.call_args.kwargs["image"] is image
    assert constructor.call_args.kwargs["install_pipx"] is True


def test_stop_waits_for_sandbox_termination_even_if_runtime_close_fails():
    sandbox = SimpleNamespace(terminate=SimpleNamespace(aio=AsyncMock()))
    environment = Environment.__new__(Environment)
    environment.deployment = SimpleNamespace(
        _sandbox=sandbox, stop=AsyncMock(side_effect=RuntimeError("close failed"))
    )

    with pytest.raises(RuntimeError, match="close failed"):
        environment.stop()

    sandbox.terminate.aio.assert_awaited_once_with(wait=True)
