import pytest
from .soft_assert import SoftAssert, SoftAssertBrokenTestError

# To store the default level and allow override via command line
_soft_assert_level_default = "broken"
_soft_assert_level_config = None


def pytest_addoption(parser):
    """Add command-line option to set default soft assert failure level."""
    group = parser.getgroup("soft-assert")
    group.addoption(
        "--soft-assert-level",
        action="store",
        default=None,  # Default will be handled by pytest_configure
        choices=("broken", "failed", "passed"),
        help="Default level for soft assertion failures: broken, failed, or passed.",
    )
    parser.addini(
        "soft_assert_level",
        type="string",
        default="broken",
        help="Default level for soft assertion failures: broken, failed, or passed (ini config).",
    )


def pytest_configure(config):
    """Register the marker and read the command-line option/ini."""
    global _soft_assert_level_config
    cmd_line_level = config.getoption("soft_assert_level")
    ini_level = config.getini("soft_assert_level")

    if cmd_line_level is not None:
        _soft_assert_level_config = cmd_line_level
    elif ini_level is not None:
        _soft_assert_level_config = ini_level
    else:
        _soft_assert_level_config = _soft_assert_level_default # Should not happen if ini has default

    config.addinivalue_line(
        "markers",
        "soft_assert_level(level): mark test to change soft assertion failure level (broken, failed, passed).",
    )


@pytest.fixture(autouse=True)
def auto_soft_assert(request):
    """
    Pytest fixture to automatically collect and assert all soft assertions
    after each test. It also clears active SoftAssert instances.
    """
    yield  # Test runs here

    # Determine the failure level for this test
    # Priority: marker > command-line/ini > default
    marker = request.node.get_closest_marker("soft_assert_level")
    level = _soft_assert_level_config or _soft_assert_level_default # Fallback

    if marker:
        if marker.args and marker.args[0] in ("broken", "failed", "passed"):
            level = marker.args[0]
        else:
            pytest.warning(
                f"Invalid soft_assert_level marker on {request.node.name}: {marker.args}. "
                f"Using default/configured level: {level}"
            )

    # Consolidate errors from all active SoftAssert instances
    all_errors_collected = []
    active_instances = SoftAssert._get_active_instances_for_test_teardown()

    for instance in active_instances:
        errors = instance.get_errors()
        if errors:
            all_errors_collected.extend(errors)
            # instance.clear_errors() # Errors are cleared by assert_all or by the class method later

    # Clear all active instances *before* potentially raising an error
    # This ensures that even if one assert_all fails, the state is clean for the next test.
    SoftAssert._clear_all_active_instances_for_test_teardown()

    if not all_errors_collected:
        return

    # Construct a consolidated error message
    num_errors = len(all_errors_collected)
    plural = "s" if num_errors > 1 else ""
    consolidated_message = f"Soft assertion(s) failed with {num_errors} error{plural} overall:\n"
    for i, error in enumerate(all_errors_collected, 1):
        consolidated_message += f"{i}. {error}\n"

    # Take action based on the determined level
    if level == "broken":
        raise SoftAssertBrokenTestError(consolidated_message.strip())
    elif level == "failed":
        pytest.fail(consolidated_message.strip(), pytrace=False)
    elif level == "passed":
        # Errors are already logged to Allure by _add_failure.
        # We can print a summary to console if desired.
        print(f"INFO: Soft assertion errors recorded but test '{request.node.name}' marked as passed:\n{consolidated_message.strip()}")
        # Optionally, attach the summary to Allure as well
        try:
            import allure
            allure.attach(consolidated_message, name="Soft Assertions Summary (Passed with Errors)", attachment_type=allure.attachment_type.TEXT)
        except ImportError:
            pass # Allure not installed or configured
        except Exception: # pragma: no cover
            pass # Catch any other allure errors
    else: # Should not happen due to choices/validation
        raise ValueError(f"Internal error: Invalid soft assertion level '{level}'.")

# Optional: Hook to customize report for SoftAssertBrokenTestError
def pytest_report_teststatus(report, config):
    if report.when == "call" and report.failed:
        if isinstance(report.longrepr, tuple) and report.longrepr[0].endswith("SoftAssertBrokenTestError"):
            return "broken", "B", ("BROKEN", {"purple": True})
    return None