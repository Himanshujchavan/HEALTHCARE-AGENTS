import json
from Agents.masterhealth import MasterHealthAgent

def test_diabetes_snapshot():
    print("Running Snapshot Test for Diabetes...")
    
    # Required snapshot values from assigments.md
    test_data = {"hba1c": 6.8, "glucose": 148, "bmi": 29, "age": 45}
    
    # Expected values (from legacy baseline)
    # Based on rule_risk: 
    # hba1c 6.8 (0.6 * 0.35) + glucose 148 (0.6 * 0.35) + bmi 29 (0.4 * 0.2) + age 45 (0.4 * 0.1)
    # = 0.21 + 0.21 + 0.08 + 0.04 = 0.54 -> 54%
    expected_score = 54.0
    expected_level = "HIGH"

    agent = MasterHealthAgent()
    result = agent.process_health_data(test_data)
    
    final = result.get("final_assessment", {})
    actual_score = final.get("score", 0) * 100 if final.get("score", 0) < 1 else final.get("score", 0)
    actual_level = final.get("risk_level")

    print(f"Input: {test_data}")
    print(f"Expected: Score {expected_score}, Level {expected_level}")
    print(f"Actual:   Score {actual_score}, Level {actual_level}")

    if abs(actual_score - expected_score) < 1 and actual_level == expected_level:
        print("✅ SNAPSHOT TEST PASSED")
        return True
    else:
        print("❌ SNAPSHOT TEST FAILED")
        return False

if __name__ == "__main__":
    try:
        success = test_diabetes_snapshot()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Error during test: {e}")
        exit(1)
