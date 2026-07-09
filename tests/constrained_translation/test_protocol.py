"""Tests for Task 1: package scaffold — import and version."""
import constrained_translation


def test_package_imports():
    """Package imports without error."""
    assert constrained_translation is not None


def test_package_version():
    """Package exposes version 0.1.0."""
    assert constrained_translation.__version__ == "0.1.0"
