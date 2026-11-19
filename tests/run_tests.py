"""Run the test-suite without pytest installed (the sandbox has no network).

Usage: python3 tests/run_tests.py
Provides a minimal `pytest` shim (approx, raises) only if real pytest is unavailable.
"""
import importlib
import math
import os
import sys
import traceback
import types

try:
    import pytest  # noqa: F401
except ImportError:
    shim = types.ModuleType("pytest")

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.e, self.rel, self.abs = expected, rel, abs

        def __eq__(self, other):
            return math.isclose(other, self.e, rel_tol=self.rel, abs_tol=self.abs)

        def __ne__(self, other):
            return not self.__eq__(other)

        def __repr__(self):
            return f"approx({self.e})"

    def approx(expected, rel=1e-6, abs=1e-12):
        return _Approx(expected, rel=rel, abs=abs)

    class raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"DID NOT RAISE {self.exc}")
            return issubclass(et, self.exc)

    shim.approx, shim.raises = approx, raises
    sys.modules["pytest"] = shim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
mods = sys.argv[1:] or ["test_ppl_estimators", "test_manuscript_numbers"]
failed = passed = 0
for name in mods:
    if not os.path.exists(os.path.join(HERE, name + ".py")):
        continue
    mod = importlib.import_module(name)
    for attr in sorted(dir(mod)):
        if attr.startswith("test_") and callable(getattr(mod, attr)):
            try:
                getattr(mod, attr)()
                passed += 1
                print(f"PASS {name}.{attr}")
            except Exception:
                failed += 1
                print(f"FAIL {name}.{attr}")
                traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
