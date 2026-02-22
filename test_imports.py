#!/usr/bin/env python3
"""Test that all imports work."""

import sys
import traceback

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    modules = [
        "auth",
        "memory",
        "research",
        "introspection",
        "commands",
        "tools",
        "scheduler",
        "data_security",
        "self_improve",
        "server",
    ]
    
    results = [test_import(m) for m in modules]
    
    print(f"\n{sum(results)}/{len(results)} modules imported successfully")
    sys.exit(0 if all(results) else 1)
