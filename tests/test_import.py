#!/usr/bin/env python3
"""Simple import test."""


def test_import():
    from audioear_standalone import AudioEarInference, save_obj, preprocess_image

    print("All imports OK")
    assert AudioEarInference is not None
    assert save_obj is not None
    assert preprocess_image is not None


if __name__ == "__main__":
    test_import()
