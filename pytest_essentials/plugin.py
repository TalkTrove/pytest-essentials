import pytest
import time
import random
from .soft_assert import SoftAssert, SoftAssertBrokenTestError

# To store the default level and allow override via command line
_soft_assert_level_default = "broken"
_soft_assert_level_config = None

# Retry plugin configuration globals
_retry_config_global_max_reruns = 0
_retry_config_global_delay = 0.0
_retry_config_global_backoff = 1.0 # Default backoff factor
_retry_config_global_only_marker = False

# Stash keys for retry state
RETRY_ITEM_STATE_KEY = pytest.StashKey()
TERMINALLY_FAILED_NODE_IDS_KEY = pytest.StashKey()


def pytest_addoption(parser):
    """Add command-line options for pytest-essentials."""
    group = parser.getgroup("soft-assert", "Soft Assertion Options")
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

    retry_group = parser.getgroup("retry", "Test Retries Options")
    retry_group.addoption(
        "--rerun",
        action="store",
        type=int,
        default=0,
        help="Number of times to rerun failed tests. Default: 0 (no reruns).",
    )
    retry_group.addoption(
        "--rerun-delay", # Changed from --delay to avoid potential clashes
        action="store",
        type=float,
        default=0.0, # Default to no delay
        help="Delay in seconds between reruns. Default: 0.0.",
    )
    retry_group.addoption(
        "--rerun-backoff",
        action="store",
        type=float,
        default=1.0, # Default to no backoff factor
        help="Backoff factor for delays between reruns (e.g., 2 for doubling delay). Default: 1.0.",
    )
    retry_group.addoption(
        "--rerun-only-marker", # Changed from --only-marker for clarity
        action="store_true",
        default=False,
        help="Only rerun tests marked with @pytest.mark.retry. Default: False.",
    )
    # It's good practice to also allow these via ini
    parser.addini("rerun", type="string", default="0", help="Default number of reruns for failed tests.")
    parser.addini("rerun_delay", type="string", default="0.0", help="Default delay in seconds between reruns.")
    parser.addini("rerun_backoff", type="string", default="1.0", help="Default backoff factor for delays.")
    parser.addini("rerun_only_marker", type="bool", default=False, help="Default for only rerunning marked tests.")


def pytest_configure(config):
    """Register markers and read command-line options/ini."""
    global _soft_assert_level_config, _retry_config_global_max_reruns, \
           _retry_config_global_delay, _retry_config_global_backoff, \
           _retry_config_global_only_marker

    # Configure Soft Assert Level
    cmd_line_soft_assert_level = config.getoption("soft_assert_level")
    ini_soft_assert_level = config.getini("soft_assert_level")

    if cmd_line_soft_assert_level is not None:
        _soft_assert_level_config = cmd_line_soft_assert_level
    elif ini_soft_assert_level is not None: # Check if it's a valid string from ini
        _soft_assert_level_config = ini_soft_assert_level
    else: # Should ideally not happen if ini has a default, but as a fallback
        _soft_assert_level_config = _soft_assert_level_default

    # Configure Retry Settings
    # Priority: CLI > INI > Defaults defined with global vars
    cli_rerun = config.getoption("rerun")
    ini_rerun_str = config.getini("rerun")
    if cli_rerun is not None and cli_rerun > 0: # CLI takes precedence if set and > 0
        _retry_config_global_max_reruns = cli_rerun
    elif ini_rerun_str: # Check if ini_rerun_str is not empty
        try:
            ini_rerun_val = int(ini_rerun_str)
            if ini_rerun_val > 0:
                 _retry_config_global_max_reruns = ini_rerun_val
        except ValueError:
            pytest.warning(f"Invalid value for 'rerun' in ini: '{ini_rerun_str}'. Using default: {_retry_config_global_max_reruns}")

    cli_delay = config.getoption("rerun_delay")
    ini_delay_str = config.getini("rerun_delay")
    if cli_delay is not None and cli_delay >= 0:
        _retry_config_global_delay = cli_delay
    elif ini_delay_str:
        try:
            _retry_config_global_delay = float(ini_delay_str)
            if _retry_config_global_delay < 0:
                pytest.warning(f"Negative 'rerun_delay' in ini: '{ini_delay_str}'. Using 0.0.")
                _retry_config_global_delay = 0.0
        except ValueError:
            pytest.warning(f"Invalid value for 'rerun_delay' in ini: '{ini_delay_str}'. Using default: {_retry_config_global_delay}")
    
    cli_backoff = config.getoption("rerun_backoff")
    ini_backoff_str = config.getini("rerun_backoff")
    if cli_backoff is not None and cli_backoff >= 1.0:
        _retry_config_global_backoff = cli_backoff
    elif ini_backoff_str:
        try:
            ini_backoff_val = float(ini_backoff_str)
            if ini_backoff_val >= 1.0:
                _retry_config_global_backoff = ini_backoff_val
            else:
                pytest.warning(f"'rerun_backoff' in ini must be >= 1.0: '{ini_backoff_str}'. Using default: {_retry_config_global_backoff}")
        except ValueError:
            pytest.warning(f"Invalid value for 'rerun_backoff' in ini: '{ini_backoff_str}'. Using default: {_retry_config_global_backoff}")

    # For boolean flags like rerun_only_marker, getoption directly gives the bool
    # For ini, addini(type="bool") handles common string bools ("true", "false", "1", "0")
    _retry_config_global_only_marker = config.getoption("rerun_only_marker")
    if not _retry_config_global_only_marker: # If CLI is false, check INI
        _retry_config_global_only_marker = config.getini("rerun_only_marker")


    config.addinivalue_line(
        "markers",
        "soft_assert_level(level): mark test to change soft assertion failure level (broken, failed, passed).",
    )
    config.addinivalue_line(
        "markers",
        "retry(n=None, delay=None, backoff=None, filter=None): mark test to be retried on failure. "
        "'n': max retries (int). "
        "'delay': initial delay in seconds (float). "
        "'backoff': multiplier for delay (float, >=1.0). "
        "'filter': an Exception type or tuple of types to retry on.",
    )

    # Initialize the set for terminally failed node IDs on config.stash
    config.stash[TERMINALLY_FAILED_NODE_IDS_KEY] = set()


def _get_retry_params_for_item(item, config):
    """
    Determines the retry parameters for a given test item, considering
    global configuration and the @pytest.mark.retry marker.
    """
    marker = item.get_closest_marker("retry")

    # Start with global defaults
    max_retries = _retry_config_global_max_reruns
    current_delay = _retry_config_global_delay
    backoff_factor = _retry_config_global_backoff
    exception_filter = None # Default to retry on any exception if retries are active

    if _retry_config_global_only_marker and not marker:
        return 0, 0.0, 1.0, None # No retries if --rerun-only-marker and no marker

    if marker:
        # Marker overrides global settings if specified
        # Marker args: n=None, delay=None, backoff=None, filter=None
        marker_n = marker.kwargs.get("n")
        marker_delay = marker.kwargs.get("delay")
        marker_backoff = marker.kwargs.get("backoff")
        marker_filter = marker.kwargs.get("filter")

        if marker_n is not None:
            try:
                max_retries = int(marker_n)
                if max_retries < 0:
                    pytest.warning(f"Invalid 'n' ({marker_n}) in @retry for {item.nodeid}. Using 0.")
                    max_retries = 0
            except (ValueError, TypeError):
                pytest.warning(f"Invalid type for 'n' ({marker_n}) in @retry for {item.nodeid}. Using global/default.")
        
        if marker_delay is not None:
            try:
                current_delay = float(marker_delay)
                if current_delay < 0:
                    pytest.warning(f"Invalid 'delay' ({marker_delay}) in @retry for {item.nodeid}. Using 0.0.")
                    current_delay = 0.0
            except (ValueError, TypeError):
                pytest.warning(f"Invalid type for 'delay' ({marker_delay}) in @retry for {item.nodeid}. Using global/default.")

        if marker_backoff is not None:
            try:
                backoff_factor = float(marker_backoff)
                if backoff_factor < 1.0: # Backoff should not reduce delay
                    pytest.warning(f"Invalid 'backoff' ({marker_backoff}) in @retry for {item.nodeid} (must be >= 1.0). Using 1.0.")
                    backoff_factor = 1.0
            except (ValueError, TypeError):
                pytest.warning(f"Invalid type for 'backoff' ({marker_backoff}) in @retry for {item.nodeid}. Using global/default.")
        
        if marker_filter is not None:
            if isinstance(marker_filter, type) and issubclass(marker_filter, BaseException):
                exception_filter = marker_filter
            elif isinstance(marker_filter, tuple) and all(isinstance(ef, type) and issubclass(ef, BaseException) for ef in marker_filter):
                exception_filter = marker_filter
            else:
                pytest.warning(
                    f"Invalid 'filter' ({marker_filter}) in @retry for {item.nodeid}. "
                    f"It must be an Exception type or a tuple of Exception types. Ignoring filter."
                )
                exception_filter = None # Fallback to no specific filter

    # If no marker and --rerun is set (and not --rerun-only-marker), use global settings
    # This case is already handled by initializing with global defaults and the _retry_config_global_only_marker check

    return max_retries, current_delay, backoff_factor, exception_filter


@pytest.hookimpl(tryfirst=True) # Run before other setup hooks
def pytest_runtest_setup(item):
    """
    Initialize retry state for the test item before it runs.
    This includes capturing the initial random state if retries are active.
    """
    if not hasattr(item, 'stash'): # Should always be present in modern pytest
        return

    max_retries, initial_delay, backoff_factor, exc_filter = _get_retry_params_for_item(item, item.config)

    # Only initialize retry state if there's a chance of retrying
    if max_retries > 0:
        item.stash[RETRY_ITEM_STATE_KEY] = {
            "attempts": 0, # Number of attempts made so far (0 means first run)
            "max_retries": max_retries,
            "current_delay": initial_delay, # This will be the delay *before* the first retry
            "backoff_factor": backoff_factor,
            "exception_filter": exc_filter,
            "original_random_state": random.getstate(), # Save initial random state
            "reports": [] # To store reports from each attempt
        }
    else:
        # Ensure key is not present if no retries, or clear if it was somehow set
        if RETRY_ITEM_STATE_KEY in item.stash:
            del item.stash[RETRY_ITEM_STATE_KEY]


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

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """
    Called to create a test report for each of setup, call, and teardown phases.
    This is where we implement the retry logic.
    """
    # We only care about the 'call' phase for retrying test execution
    if call.when != 'call':
        return None # Let pytest handle report for setup/teardown as usual

    retry_state = item.stash.get(RETRY_ITEM_STATE_KEY)
    report = pytest.TestReport.from_call(call) # Get the default report for this call

    # Store all 'call' reports in the retry_state if it exists
    if retry_state:
        retry_state.setdefault('call_reports', []).append(report)

    if not report.failed or not retry_state:
        # If test passed, or no retry state, nothing to do here for retries
        # If it passed and had retries, we'll note it in the final report attributes
        if report.passed and retry_state and retry_state["attempts"] > 0:
            report.num_attempts = retry_state["attempts"] + 1 # attempts is num_retries, so +1 for total
        return report # Return the created report

    # At this point, test failed (report.failed is True) and retry_state exists.

    # Check if this specific exception type should be retried
    should_retry_this_exception = True
    if retry_state["exception_filter"]:
        if not call.excinfo or not isinstance(call.excinfo.value, retry_state["exception_filter"]):
            should_retry_this_exception = False

    terminally_failed_ids = item.config.stash.get(TERMINALLY_FAILED_NODE_IDS_KEY, set())
    if item.nodeid in terminally_failed_ids:
        should_retry_this_exception = False # Do not retry terminally failed tests

    # Check if we have retries left and the exception type matches
    if retry_state["attempts"] < retry_state["max_retries"] and should_retry_this_exception:
        if item.session.shouldstop: # Check for -x or other stop conditions
             # If pytest is stopping, don't attempt further retries
            pass # Let the failure propagate as final
        else:
            # Yes, we should retry this test
            retry_state["attempts"] += 1

            # Perform delay
            actual_delay = retry_state["current_delay"]
            if actual_delay > 0:
                time.sleep(actual_delay)
            
            # Update delay for next potential retry using backoff
            retry_state["current_delay"] *= retry_state["backoff_factor"]

            # Restore random state
            random.setstate(retry_state["original_random_state"])

            # Modify the current report to indicate a "rerun"
            report.outcome = "rerun" # Custom outcome
            report.longrepr = None   # Clear detailed error for this intermediate "rerun" report
                                     # The actual failure is in retry_state['call_reports']
            
            # Add custom attribute for reporting
            report.rerun_attempt = retry_state["attempts"]


            # Reschedule the item for another run.
            # This involves adding it back to the session's processing stack.
            # Pytest's internal `SetupState().stack` is what drives test execution.
            # We need to ensure the item is re-processed fully (setup, call, teardown).
            # Note: This is a somewhat low-level manipulation.
            # Ensure finalizers from the current failed attempt are run.
            # Pytest should handle this as part of its normal item execution flow
            # when the current item finishes its teardown, before the rescheduled one starts.
            
            # Clear existing reports for the item to avoid confusion on rerun
            # item.reports = [r for r in item.reports if r.when != 'call'] # Risky, might clear too much
                                                                        # Better to let pytest manage reports list
                                                                        # and we just influence the current one.

            # The crucial part: make pytest run this item again.
            # By appending to the stack, it should be picked up by the main loop.
            item.session._setupstate.stack.append(item)
            
            # We might need to signal that the current test "call" phase is over,
            # but will be retried, so it doesn't get counted as a final failure yet
            # by things like -x. Setting outcome to "rerun" should help.
            # If -x is active, item.session.shouldstop would be true, handled above.

            return report # Return the modified "rerun" report

    # If we are here, it's a final failure (no more retries, or exception didn't match, or test passed)
    # Ensure the report reflects the *last actual failure* if retries occurred.
    if report.failed and retry_state and retry_state["call_reports"]:
        # Find the last report that was a genuine failure (not a "rerun" we might have made)
        last_actual_failure_report = None
        for r in reversed(retry_state["call_reports"]):
            if r.failed and r.outcome != "rerun": # Check original outcome before we might have changed it
                 # A bit tricky as we modify 'report' in place.
                 # Let's assume the 'call.excinfo' on the *current* call object is the one for the last attempt.
                 pass # The current 'report' object is already based on the last 'call'

        # Add retry summary to the longrepr of the final failed report
        num_retries_done = retry_state["attempts"]
        if num_retries_done > 0:
            original_longrepr_text = report.longreprtext
            retry_summary = f"\n\n[pytest-essentials] Test failed after {num_retries_done} retries."
            
            # Collect details of exceptions from each attempt for the final report
            attempt_details = []
            for i, r_attempt in enumerate(retry_state["call_reports"]):
                if r_attempt.failed and hasattr(r_attempt, 'longreprtext') and r_attempt.longreprtext:
                    # Only include if it was a failure and has a longrepr
                    # Avoid including our own "rerun" reports if they somehow got here with longrepr
                    if hasattr(r_attempt, 'rerun_attempt'): # This was a retry attempt we marked
                        attempt_details.append(f"  Attempt {i+1} (retry {i}):\n{r_attempt.longreprtext}")
                    elif i == len(retry_state["call_reports"]) -1 : # Last attempt, which is the current 'report'
                         attempt_details.append(f"  Attempt {i+1} (final):\n{original_longrepr_text}")


            if attempt_details: # Only add if there are actual failure details
                 # Use the longrepr from the very last attempt for the main report body
                report.longreprtext = original_longrepr_text + retry_summary
                # Optionally, could append all attempt details:
                # report.longreprtext += "\nFailure details per attempt:\n" + "\n".join(attempt_details)

            report.num_attempts = num_retries_done + 1


    return report # Return the (potentially final) report


# Combined hook to customize report for SoftAssertBrokenTestError and Retries
def pytest_report_teststatus(report, config):
    if report.when == "call":
        if hasattr(report, "rerun_attempt") and report.outcome == "rerun":
            return "rerun", "R", ("RERUNNING", {"yellow": True})
        if report.failed:
            if isinstance(getattr(report.longrepr, 'reprcrash', None), SoftAssertBrokenTestError) or \
               (isinstance(report.longrepr, tuple) and report.longrepr[0].endswith("SoftAssertBrokenTestError")): # Compatibility
                return "broken", "B", ("BROKEN", {"purple": True})
            # If it failed after retries, we could use a different status, but 'failed' is standard.
            # The summary table will show retry counts.
            # Default pytest 'failed' status will be used if not BROKEN or RERUNNING.
    return None # Default handling


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Add a summary of retried tests to the terminal output.
    """
    tr = terminalreporter
    retried_tests_info = []
    
    # Iterate through all collected reports by status
    # We are interested in tests that have the 'num_attempts' attribute and it's > 1
    for status_category in tr.stats: # e.g., 'passed', 'failed', 'skipped', 'broken', 'rerun'
        if status_category == 'rerun': # These are intermediate reports, not final outcomes
            continue
            
        category_reports = tr.stats[status_category]
        for report in category_reports:
            if report.when == 'call' and hasattr(report, 'num_attempts') and report.num_attempts > 1:
                retried_tests_info.append({
                    'nodeid': report.nodeid,
                    'attempts': report.num_attempts,
                    'status': report.outcome.upper() # PASSED, FAILED, BROKEN etc.
                })

    if retried_tests_info:
        tr.write_sep("=", "Retry Summary", bold=True)
        for info in retried_tests_info:
            # Ensure status is a string, sometimes it might be an object with outcome
            status_str = info['status']
            if not isinstance(status_str, str):
                status_str = str(status_str) # Fallback

            tr.write_line(f"{info['nodeid']}: {status_str} after {info['attempts']} attempts")
        tr.write_line("") # Extra line for spacing