from __future__ import annotations

import subprocess
import sys

import pytest

import aweagent.core.condenser.terminus_2 as terminus_2_module
from aweagent.core.condenser import build_condenser
from aweagent.core.condenser.tool_result_omission import ToolResultOmissionCondenser
from aweagent.core.condenser.truncation import TruncationCondenser
from aweagent.core.config.schema import CondenserConfig


@pytest.mark.parametrize(
    ("condenser_type", "expected_type"),
    [
        ("none", None),
        ("truncation", TruncationCondenser),
        ("tool_result_omission", ToolResultOmissionCondenser),
    ],
)
def test_existing_condenser_types_are_unchanged(
    condenser_type: str,
    expected_type: type | None,
) -> None:
    condenser = build_condenser(CondenserConfig(type=condenser_type))

    if expected_type is None:
        assert condenser is None
    else:
        assert isinstance(condenser, expected_type)


def test_package_import_does_not_eagerly_load_terminus_module() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import aweagent.core.condenser\n"
                "assert 'aweagent.core.condenser.terminus_2' not in sys.modules\n"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_terminus_factory_forwards_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class FakeTerminus2Condenser:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        terminus_2_module,
        "Terminus2Condenser",
        FakeTerminus2Condenser,
    )

    build_condenser(
        CondenserConfig(type="terminus_2", tokenizer_path=str(tmp_path))
    )

    assert captured == {
        "enable_summarize": True,
        "proactive_threshold": 8000,
        "recovery_target_free_tokens": 4000,
        "tokenizer_path": str(tmp_path),
    }
