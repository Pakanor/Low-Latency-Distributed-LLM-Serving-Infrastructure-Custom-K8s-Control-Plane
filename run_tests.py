#!/usr/bin/env python3
import sys
import os
import subprocess

os.chdir('/home/roberto/llm-infra')
sys.path.insert(0, '/home/roberto/llm-infra')

tests = [
        "tests/test_sequence.py",
        "tests/test_kv_cache_manager.py",
        "tests/test_scheduler.py",
        "tests/test_llm_engine.py",
]

print("=" * 60)
print("RUNNING TESTS")
print("=" * 60)

all_passed = True
for test_file in tests:
    print(f"\n[TEST] {test_file}")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = '/home/roberto/llm-infra'
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            cwd='/home/roberto/llm-infra'
        )
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout)
            print(f"✓ PASSED")
        else:
            print(f"✗ FAILED")
            if result.stderr:
                print(result.stderr)
            all_passed = False
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT")
        all_passed = False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED")
print("=" * 60)

sys.exit(0 if all_passed else 1)
