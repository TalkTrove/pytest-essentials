import pytest
from pytest_essentials import SoftAssert, SoftAssertBrokenTestError

# You would typically have a shared SoftAssert instance or use the class methods directly
# For these examples, we'll often create a new one or rely on the fixture's handling of class-level instances.

class TestSoftAssertions:

    def test_no_failures_instance(self):
        sa = SoftAssert()
        sa.assert_equal(1, 1, "Check one is one")
        sa.assert_true(True, "Check True is True")
        # sa.assert_all() # Implicitly called by the auto_soft_assert fixture

    def test_no_failures_class_methods_directly(self):
        # Note: Using SoftAssert class methods directly like this relies on the
        # fixture to manage a default/implicit instance or handle static collection.
        # The current implementation of SoftAssert creates an instance internally if used this way,
        # and the fixture will pick up all such instances.
        # However, for clarity, instantiating is often better.
        # This test is more to show the fixture handles multiple instances if they were created.
        # Let's assume for this pattern, a default instance is implicitly managed or
        # we are testing the fixture's ability to catch multiple independent assertions.
        # For a more robust direct class method usage, SoftAssert would need static methods
        # that manage a default internal instance.
        # The current SoftAssert() creates an instance and adds it to _active_instances.
        # So, each call below creates a new SoftAssert object.
        SoftAssert().assert_equal(5, 5, "Item A")
        SoftAssert().assert_is_not_none("hello", "Item B")
        # assert_all() is implicitly called by the fixture

    def test_single_equal_failure_default_broken(self):
        sa = SoftAssert()
        sa.assert_equal(1, 2, "Error: Value mismatch for 1 and 2")
        # Expect SoftAssertBrokenTestError due to fixture (default level)

    def test_multiple_failures_default_broken(self):
        sa = SoftAssert()
        sa.assert_equal(1, 2, "Error 1: Value mismatch")
        sa.assert_true(False, "Error 2: Condition was False")
        sa.assert_in("a", "bcd", "Error 3: 'a' not in 'bcd'")
        # Expect SoftAssertBrokenTestError

    @pytest.mark.soft_assert_level("failed")
    def test_single_false_failure_marked_failed(self):
        sa = SoftAssert()
        sa.assert_false(True, "This should fail the test directly")
        # Expect pytest.fail() due to marker

    @pytest.mark.soft_assert_level("failed")
    def test_multiple_failures_marked_failed(self):
        sa = SoftAssert()
        sa.assert_is_none("not none", "Error A: Should be None")
        sa.assert_not_in(5, [1, 2, 3, 4, 5], "Error B: 5 should not be in list")
        # Expect pytest.fail()

    @pytest.mark.soft_assert_level("passed")
    def test_single_true_failure_marked_passed(self):
        sa = SoftAssert()
        sa.assert_true(False, "This error will be logged, but test passes")
        # Expect test to pass, error logged (e.g., to Allure and console)

    @pytest.mark.soft_assert_level("passed")
    def test_multiple_failures_marked_passed(self):
        sa = SoftAssert()
        sa.assert_equal("x", "y", "Mismatch x and y")
        sa.assert_is_not_none(None, "Object was None")
        # Expect test to pass, errors logged

    def test_mixed_assertions_with_one_failure(self):
        sa = SoftAssert()
        sa.assert_equal(10, 10, "Correct: 10 is 10")
        sa.assert_false(1 == 2, "Correct: 1 is not equal to 2")
        sa.assert_in("key", {"key": "value"}, "Correct: key in dict")
        sa.assert_equal("apple", "orange", "Failure: apple vs orange") # This will fail
        sa.assert_true("some string".startswith("some"), "Correct: string starts with some")
        # Expect SoftAssertBrokenTestError (default level)

    def test_chaining_assertions(self):
        sa = SoftAssert()
        (sa.assert_equal(1, 1, "Chain 1")
           .assert_true(True, "Chain 2")
           .assert_false(False, "Chain 3")
           .assert_equal(1, 2, "Chain 4 - Failure") # This will cause a failure
           .assert_in("a", "abc", "Chain 5"))
        # Expect SoftAssertBrokenTestError

    def test_no_soft_assert_calls(self):
        # This test makes no calls to SoftAssert
        assert True # Standard pytest assertion
        # Expect to pass cleanly

    def test_soft_assert_instance_reuse_with_clear(self):
        sa = SoftAssert()
        sa.assert_equal(1, 2, "Initial failure")
        # If we were to call sa.assert_all() here, it would raise.
        # The fixture will call it at the end.
        # If we want to clear errors mid-test (less common with auto-fixture):
        # sa.clear_errors()
        # sa.assert_equal(3, 3, "This would now be the only error if cleared")
        # For this test, we'll let the fixture handle the "Initial failure".
        # Expect SoftAssertBrokenTestError

    # Example of how a user might use assert_all manually if they disable autouse or need intermediate checks
    # For this to work as expected without the autouse fixture, the fixture would need to be non-autouse
    # or this test would need to manage its SoftAssert instances entirely separately.
    # Given the autouse fixture, this manual assert_all will run, and then the fixture will run again.
    def test_manual_assert_all_then_fixture(self):
        sa = SoftAssert()
        sa.assert_equal(100, 101, "Manual check failure")
        with pytest.raises(SoftAssertBrokenTestError, match="Manual check failure"):
            sa.assert_all() # This will raise and clear errors for 'sa'

        # Because sa.assert_all() was called and cleared its errors,
        # when the auto_soft_assert fixture runs for *this specific instance 'sa'*,
        # it will find no errors *for 'sa'*.
        # If other SoftAssert instances were used and not cleared, fixture would catch them.
        # This also means _active_instances needs careful handling if assert_all clears the instance from it.
        # Current SoftAssert.assert_all() only clears self._errors, not from _active_instances.
        # The fixture clears _active_instances globally after collecting from all.
        # So, this test should pass the fixture's check because 'sa' errors are cleared.
        # However, the test itself "fails" at the pytest.raises if not for the raise.

    def test_two_soft_assert_instances(self):
        sa1 = SoftAssert()
        sa2 = SoftAssert()

        sa1.assert_equal(1, 2, "SA1 Failure 1")
        sa2.assert_true(False, "SA2 Failure 1")
        sa1.assert_in("x", "abc", "SA1 Failure 2")

        # The auto_soft_assert fixture should collect errors from both sa1 and sa2.
        # Expect SoftAssertBrokenTestError with 3 errors.

    @pytest.mark.soft_assert_level("failed")
    def test_two_soft_assert_instances_level_failed(self):
        sa1 = SoftAssert()
        sa2 = SoftAssert()

        sa1.assert_equal(1, 2, "SA1 Failure (level:failed)")
        sa2.assert_true(False, "SA2 Failure (level:failed)")
        # Expect pytest.fail() with 2 errors.

    @pytest.mark.soft_assert_level("passed")
    def test_two_soft_assert_instances_level_passed(self):
        sa1 = SoftAssert()
        sa2 = SoftAssert()

        sa1.assert_equal(1, 2, "SA1 Failure (level:passed)")
        sa2.assert_true(False, "SA2 Failure (level:passed)")
        # Expect test to pass, but 2 errors logged.

    def test_allure_attachment_on_failure(self):
        # This test assumes Allure is installed and configured if you want to see attachments.
        # The plugin itself handles Allure being optional.
        sa = SoftAssert()
        sa.assert_equal("allure", "report", "Check Allure attachment")
        # If Allure is active, an attachment "Assertion Failure" should be made.
        # Expect SoftAssertBrokenTestError.
        # To verify Allure: run pytest with --alluredir and then generate the report.
        # e.g., pytest --alluredir=allure-results && allure generate allure-results && allure open

    # Test to ensure SoftAssertBrokenTestError is correctly raised for default 'broken'
    def test_broken_exception_type(self):
        sa = SoftAssert()
        sa.assert_equal(1, 0, "Trigger broken")
        # The fixture will handle this. To test the exception directly:
        # with pytest.raises(SoftAssertBrokenTestError, match="Trigger broken"):
        #     sa.assert_all() # This would be a manual check

    # Test to ensure pytest.fail is called for 'failed'
    # This is harder to test directly for the fixture's behavior without deeper pytest introspection
    # or by checking the report outcome. The fixture uses pytest.fail().

    # Test to ensure test passes for 'passed' despite errors
    # Similar to above, best verified by checking test report outcome.
    # The fixture prints a message and doesn't raise for 'passed'.