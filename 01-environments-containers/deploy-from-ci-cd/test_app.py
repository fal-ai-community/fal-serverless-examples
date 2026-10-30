"""Local endpoint and input-contract tests without credentials or remote compute."""

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

_spec = importlib.util.spec_from_file_location(
    "ci_cd_demo_app", Path(__file__).parent / "app.py"
)
assert _spec is not None and _spec.loader is not None
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)


def test_greet():
    app = app_module.CICDDemo(_allow_init=True)
    response = app.greet(app_module.Input(name="Ada Lovelace"))
    assert response.model_dump() == {"message": "Hello, Ada Lovelace!"}


def test_name_is_required():
    with pytest.raises(ValidationError):
        app_module.Input.model_validate({})


def test_name_must_be_a_string():
    with pytest.raises(ValidationError):
        app_module.Input.model_validate({"name": 123})
