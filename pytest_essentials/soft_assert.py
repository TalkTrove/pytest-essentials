try:
    import allure
except ImportError:
    allure = None

class SoftAssertBrokenTestError(Exception):
    """Custom exception for soft assertion failures to mark tests as broken."""
    pass

class SoftAssert:
    _active_instances = []  # Class variable to track instances

    def __init__(self):
        self._errors = []
        # Only add if not already present
        if self not in SoftAssert._active_instances:
            SoftAssert._active_instances.append(self)

    @classmethod
    def _clear_all_active_instances_for_test_teardown(cls):
        """Clears all tracked instances. Should be called by a test teardown fixture."""
        cls._active_instances = []

    @classmethod
    def _get_active_instances_for_test_teardown(cls):
        """Gets a copy of active instances. Should be called by a test teardown fixture."""
        return list(cls._active_instances)

    def _add_failure(self, message):
        self._errors.append(message)
        if allure:
            try:
                allure.attach(message, name="Assertion Failure", attachment_type=allure.attachment_type.TEXT)
            except Exception:  # pragma: no cover
                # Allure might be imported but not configured,
                # or other Allure-specific errors could occur.
                pass

    def assert_equal(self, actual, expected, message=""):
        if actual != expected:
            failure_message = f"{message} Expected '{expected}', but got '{actual}'."
            self._add_failure(failure_message)
        return self # Allow chaining

    def assert_true(self, condition, message=""):
        if not condition:
            failure_message = f"{message} Expected condition to be True, but it was False."
            self._add_failure(failure_message)
        return self

    def assert_false(self, condition, message=""):
        if condition:
            failure_message = f"{message} Expected condition to be False, but it was True."
            self._add_failure(failure_message)
        return self

    def assert_in(self, member, container, message=""):
        if member not in container:
            failure_message = f"{message} Expected '{member}' to be in '{container}', but it was not."
            self._add_failure(failure_message)
        return self

    def assert_not_in(self, member, container, message=""):
        if member in container:
            failure_message = f"{message} Expected '{member}' not to be in '{container}', but it was."
            self._add_failure(failure_message)
        return self

    def assert_is_none(self, obj, message=""):
        if obj is not None:
            failure_message = f"{message} Expected object to be None, but it was '{obj}'."
            self._add_failure(failure_message)
        return self

    def assert_is_not_none(self, obj, message=""):
        if obj is None:
            failure_message = f"{message} Expected object not to be None, but it was."
            self._add_failure(failure_message)
        return self

    def get_errors(self):
        """Returns a list of all collected error messages."""
        return list(self._errors)

    def clear_errors(self):
        """Clears all collected error messages for this instance."""
        self._errors = []

    def assert_all(self, level="broken"):
        """
        Checks all collected assertions.
        If any failures were recorded, raises an exception to mark the test
        according to the specified level.
        Clears errors for this instance after checking.
        """
        if not self._errors:
            return

        all_errors_message = f"Soft assertion failed with {len(self._errors)} error(s):\n"
        for i, error in enumerate(self._errors, 1):
            all_errors_message += f"{i}. {error}\n"
        
        # Clear errors for this specific instance before raising
        # This is important if assert_all is called multiple times or if the instance is reused.
        # However, the primary mechanism for clearing will be the class-level teardown.
        current_errors = list(self._errors) # Keep a copy for the exception
        self.clear_errors()

        # Reconstruct the message for the exception with the errors that were present at call time
        exception_message = f"Soft assertion failed with {len(current_errors)} error(s):\n"
        for i, error in enumerate(current_errors, 1):
            exception_message += f"{i}. {error}\n"

        if level == "broken":
            raise SoftAssertBrokenTestError(exception_message.strip())
        elif level == "failed":
            # We will use pytest.fail in the fixture for 'failed'
            # For direct assert_all call, we can raise a generic error or SoftAssertBrokenTestError
            # as a placeholder, actual handling will be in conftest.
            raise AssertionError(exception_message.strip()) 
        elif level == "passed":
            # If level is 'passed', we don't raise an exception even if there are errors.
            # The errors would have been logged to Allure.
            print(f"Soft assertion errors recorded but test marked as passed:\n{exception_message.strip()}")
            return
        else:
            raise ValueError(f"Invalid assertion level: {level}. Must be 'broken', 'failed', or 'passed'.")