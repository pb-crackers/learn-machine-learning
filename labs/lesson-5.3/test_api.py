"""
Lab 5.3 — API Test Script
===========================
Tests the prediction API for correctness, validation, and determinism.

Usage:
    1. Start the server:  uvicorn starter:app --host 0.0.0.0 --port 8000
    2. Run this script:   python test_api.py

All tests print PASS or FAIL. The script exits with code 0 if all pass,
or code 1 if any fail.
"""

import sys
import requests

BASE_URL = "http://localhost:8000"

# A valid sample input. Customize this to match YOUR dataset's features.
VALID_INPUT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}

passed = 0
failed = 0


def check(name, condition, detail=""):
    """Record a test result."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# -------------------------------------------------------------------------
# Test 1: Health Check
# -------------------------------------------------------------------------
print("\n--- Test 1: Health Check ---")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    check("Status code is 200", resp.status_code == 200, f"got {resp.status_code}")
    data = resp.json()
    check("Status is 'healthy'", data.get("status") == "healthy")
    check("Model is loaded", data.get("model_loaded") is True)
except requests.ConnectionError:
    print("  FAIL: Could not connect to server. Is it running on port 8000?")
    sys.exit(1)


# -------------------------------------------------------------------------
# Test 2: Valid Prediction
# -------------------------------------------------------------------------
print("\n--- Test 2: Valid Prediction ---")
resp = requests.post(f"{BASE_URL}/predict", json=VALID_INPUT, timeout=10)
check("Status code is 200", resp.status_code == 200, f"got {resp.status_code}")

data = resp.json()
check("Response has 'prediction' field", "prediction" in data)
check("Response has 'probability' field", "probability" in data)
check("Prediction is 0 or 1", data.get("prediction") in [0, 1],
      f"got {data.get('prediction')}")
check("Probability between 0 and 1",
      0.0 <= data.get("probability", -1) <= 1.0,
      f"got {data.get('probability')}")

print(f"  (Prediction: {data.get('prediction')}, Probability: {data.get('probability')})")


# -------------------------------------------------------------------------
# Test 3: Missing Required Field
# -------------------------------------------------------------------------
print("\n--- Test 3: Missing Required Field ---")
incomplete = {"age": 55}  # Missing all other fields
resp = requests.post(f"{BASE_URL}/predict", json=incomplete, timeout=10)
check("Rejects with 422", resp.status_code == 422, f"got {resp.status_code}")


# -------------------------------------------------------------------------
# Test 4: Out-of-Range Value
# -------------------------------------------------------------------------
print("\n--- Test 4: Out-of-Range Value ---")
bad_input = VALID_INPUT.copy()
bad_input["age"] = -10  # Negative age should be rejected
resp = requests.post(f"{BASE_URL}/predict", json=bad_input, timeout=10)
check("Rejects negative age with 422", resp.status_code == 422, f"got {resp.status_code}")


# -------------------------------------------------------------------------
# Test 5: Wrong Type
# -------------------------------------------------------------------------
print("\n--- Test 5: Wrong Type ---")
bad_type = VALID_INPUT.copy()
bad_type["age"] = "not_a_number"
resp = requests.post(f"{BASE_URL}/predict", json=bad_type, timeout=10)
check("Rejects string age with 422", resp.status_code == 422, f"got {resp.status_code}")


# -------------------------------------------------------------------------
# Test 6: Empty Body
# -------------------------------------------------------------------------
print("\n--- Test 6: Empty Body ---")
resp = requests.post(f"{BASE_URL}/predict", json={}, timeout=10)
check("Rejects empty body with 422", resp.status_code == 422, f"got {resp.status_code}")


# -------------------------------------------------------------------------
# Test 7: Deterministic Predictions
# -------------------------------------------------------------------------
print("\n--- Test 7: Deterministic Predictions ---")
results = []
for i in range(5):
    resp = requests.post(f"{BASE_URL}/predict", json=VALID_INPUT, timeout=10)
    results.append(resp.json().get("probability"))

unique_probs = set(results)
check("Same input gives same output (5 calls)",
      len(unique_probs) == 1,
      f"got {len(unique_probs)} distinct values: {unique_probs}")


# -------------------------------------------------------------------------
# Test 8: Boundary Values
# -------------------------------------------------------------------------
print("\n--- Test 8: Boundary Values ---")
boundary_input = VALID_INPUT.copy()
boundary_input["age"] = 0  # Minimum valid age
resp = requests.post(f"{BASE_URL}/predict", json=boundary_input, timeout=10)
check("Accepts age=0 (boundary)", resp.status_code == 200, f"got {resp.status_code}")

boundary_input["age"] = 120  # Maximum valid age
resp = requests.post(f"{BASE_URL}/predict", json=boundary_input, timeout=10)
check("Accepts age=120 (boundary)", resp.status_code == 200, f"got {resp.status_code}")


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "=" * 50)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")

if failed == 0:
    print("All tests passed.")
    sys.exit(0)
else:
    print("Some tests failed. Review the output above.")
    sys.exit(1)
