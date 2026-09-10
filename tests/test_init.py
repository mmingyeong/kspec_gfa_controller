# tests/test_init.py
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


UNDER_TEST_PACKAGE = "kspec_gfa_controller__init_under_test"
EXPORTS = {
    "GFAActions": "gfa_actions",
    "GFAEnvironment": "gfa_environment",
    "GFAGuider": "gfa_guider",
}


def _find_package_init() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "src" / "kspec_gfa_controller" / "__init__.py",
        repo_root / "kspec_gfa_controller" / "__init__.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "kspec_gfa_controller/__init__.py not found. tried:\n"
        + "\n".join(str(candidate) for candidate in candidates)
    )


@pytest.fixture
def isolated_package(monkeypatch):
    init_path = _find_package_init()
    expected_classes = {}

    for class_name, module_name in EXPORTS.items():
        fake_module = types.ModuleType(f"{UNDER_TEST_PACKAGE}.{module_name}")
        fake_class = type(class_name, (), {})
        setattr(fake_module, class_name, fake_class)
        monkeypatch.setitem(
            sys.modules,
            f"{UNDER_TEST_PACKAGE}.{module_name}",
            fake_module,
        )
        expected_classes[class_name] = fake_class

    spec = spec_from_file_location(
        UNDER_TEST_PACKAGE,
        str(init_path),
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {init_path}")

    package = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, UNDER_TEST_PACKAGE, package)
    spec.loader.exec_module(package)
    return package, expected_classes


def test_dunder_all_exports_expected_names(isolated_package):
    package, _ = isolated_package

    assert package.__all__ == ["GFAActions", "GFAEnvironment", "GFAGuider"]


@pytest.mark.parametrize("name", list(EXPORTS))
def test_getattr_executes_each_lazy_import_branch(isolated_package, name):
    package, expected_classes = isolated_package

    resolved = package.__getattr__(name)

    assert resolved is expected_classes[name]
    assert isinstance(resolved, type)


@pytest.mark.parametrize("name", ["DoesNotExist", "", "gfa_actions"])
def test_getattr_unknown_name_raises_precise_attribute_error(
    isolated_package, name
):
    package, _ = isolated_package

    with pytest.raises(AttributeError) as exc_info:
        package.__getattr__(name)

    message = str(exc_info.value)
    assert repr(name) in message
    assert package.__name__ in message


def test_builtin_getattr_uses_module_lazy_getattr(isolated_package):
    package, expected_classes = isolated_package

    resolved = getattr(package, "GFAActions")

    assert resolved is expected_classes["GFAActions"]
