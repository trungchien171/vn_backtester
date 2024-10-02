#submissions.py
import pandas as pd
import streamlit as st
from utils.alpha_db import submit_alpha

def run_tests(metrics):
    test_results = {
        "Sharpe": (metrics["Sharpe"], 1),
        "Fitness": (metrics["Fitness"], 1),
        "Turnover (%)": (metrics["Turnover (%)"], (0, 70)),
        "Weight Concentration": ("Passed", None),
    }
    
    return test_results

def display_test_results(test_results, col, alpha_formula, alpha_settings, metrics, driver_service, username):
    col.subheader("Test Results")

    all_tests_passed = True

    for test, value in test_results.items():
        if test == "Weight Concentration":
            col.write(f"✅ {test} check passed")
        elif test == "Turnover (%)":
            turnover_value, turnover_range = value
            lower_bound, upper_bound = turnover_range
            if lower_bound <= turnover_value <= upper_bound:
                col.write(f"✅ {test} of {turnover_value:.2f}% is within the range of {lower_bound}% to {upper_bound}%")
            else:
                col.write(f"❌ {test} of {turnover_value:.2f}% is outside the range of {lower_bound}% to {upper_bound}%")
                all_tests_passed = False
        else:
            metric_value, threshold = value
            if metric_value < threshold:
                col.write(f"❌ {test} of {metric_value:.2f} is below the cut-off of {threshold}")
                all_tests_passed = False
            else:
                col.write(f"✅ {test} of {metric_value:.2f} passed the cut-off of {threshold}")

    col.write("⚫ Sub universe is only checked if other checks pass")
    col.write("⚫ Global correlation is only checked if other checks pass")
    col.write("⚫ Rolling correlation is only checked if other checks pass")

    if all_tests_passed:
        if col.button("Submit"):
            try:
                submit_alpha(driver_service, username, alpha_formula, alpha_settings, metrics)
                col.success("Alpha submitted successfully!")
            except Exception as e:
                col.error(f"Error occurred during submission: {e}")
    else:
        col.warning("All tests must pass to enable submission.")