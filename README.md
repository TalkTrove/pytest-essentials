# Pytest Essentials

A Pytest plugin providing essential utilities to enhance your testing workflow, starting with a robust Soft Assert mechanism.

## Features

*   **Soft Assertions**: Collect multiple assertion failures within a single test without stopping on the first failure.
    *   Automatically checks all collected assertions at the end of each test.
    *   Integrates with Allure reports to attach assertion failure details.
    *   Configurable failure behavior per test or globally:
        *   `broken`: Mark test as "broken" (default, uses a custom `SoftAssertBrokenTestError`).
        *   `failed`: Mark test as "failed" (uses `pytest.fail()`).
        *   `passed`: Log assertion errors but allow the test to pass.

## Installation

You can install `pytest-essentials` using pip:

```bash
pip install pytest-essentials
```

If you want to use the Allure reporting integration for soft assertion details, also install `allure-pytest`:

```bash
pip install pytest-essentials[allure]
# or
pip install pytest-essentials allure-pytest
```

The plugin will be automatically discovered by Pytest once installed.

## Usage

### Soft Assertions

Import the `SoftAssert` class from `pytest_essentials`. The plugin automatically provides a fixture that will call `assert_all()` on any `SoftAssert` instances used during a test.

```python
from pytest_essentials import SoftAssert
import pytest

class TestMyFeature:
    def test_example_with_soft_asserts(self):
        sa = SoftAssert()
        sa.assert_equal(1, 1, "Message for first check")
        sa.assert_true(False, "This condition should have been true")
        sa.assert_in("key", {"other_key": "value"}, "Checking for key in dict")
        sa.assert_equal("actual", "expected", "Comparing two strings")

        # By default, if any of the above assertions fail, the test will be marked 'broken'.
        # All failures will be reported.

    @pytest.mark.soft_assert_level("failed")
    def test_soft_asserts_marked_failed(self):
        sa = SoftAssert()
        sa.assert_equal(1, 2, "This will cause a failure")
        sa.assert_true(True, "This one is okay")
        # This test will be marked as 'failed' if assert_equal(1,2) fails.

    @pytest.mark.soft_assert_level("passed")
    def test_soft_asserts_marked_passed(self):
        sa = SoftAssert()
        sa.assert_is_none("some_value", "This value should be None")
        # This test will pass, but the assertion error for assert_is_none
        # will be logged (e.g., to console and Allure if configured).
```

### Configuration

You can set the default failure level for soft assertions globally:

**1. Command-line option:**

```bash
pytest --soft-assert-level=failed
```

**2. `pytest.ini` file:**

```ini
[pytest]
soft_assert_level = failed ; broken, failed, or passed
```

The priority for determining the failure level is:
1.  `@pytest.mark.soft_assert_level("level")` marker on the test function.
2.  `--soft-assert-level` command-line option.
3.  `soft_assert_level` in `pytest.ini`.
4.  Default built-in level (`broken`).


## Contributing

Contributions are welcome! Please open an issue or submit a pull request. (Further details to be added)

## License

This project is licensed under the MIT License. (Assumed, can be changed in `pyproject.toml`)