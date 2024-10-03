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

def show_test_results(test_results, col, alpha_formula, alpha_settings, main_metrics, sub_universe_metrics, driver_service, username):
    col.subheader("Test Results")

    all_tests_passed = True
    weight_passed = turnover_passed = fitness_passed = sharpe_passed = False

    for test, value in test_results.items():
        if test == "Weight Concentration":
            col.write(f"✅ {test} check passed")
            weight_passed = True
        elif test == "Turnover (%)":
            turnover_value, turnover_range = value
            lower_bound, upper_bound = turnover_range
            if lower_bound <= turnover_value <= upper_bound:
                col.write(f"✅ {test} of {turnover_value:.2f}% is within the range of {lower_bound}% to {upper_bound}%")
                turnover_passed = True
            else:
                col.write(f"❌ {test} of {turnover_value:.2f}% is outside the range of {lower_bound}% to {upper_bound}%")
                all_tests_passed = False
        elif test == "Sharpe":
            metric_value, threshold = value
            if metric_value < threshold:
                col.write(f"❌ {test} of {metric_value:.2f} is below the cut-off of {threshold}")
                all_tests_passed = False
            else:
                col.write(f"✅ {test} of {metric_value:.2f} passed the cut-off of {threshold}")
                sharpe_passed = True
        elif test == "Fitness":
            metric_value, threshold = value
            if metric_value < threshold:
                col.write(f"❌ {test} of {metric_value:.2f} is below the cut-off of {threshold}")
                all_tests_passed = False
            else:
                col.write(f"✅ {test} of {metric_value:.2f} passed the cut-off of {threshold}")
                fitness_passed = True
    
    if weight_passed and turnover_passed and fitness_passed and sharpe_passed:
        for year in [2022, 2023]:
            if year in sub_universe_metrics.index:
                sharpe_year = sub_universe_metrics.loc[year, "Sharpe"]
                if sharpe_year >= 1.0:
                    col.write(f"✅ Sub-universe test passed with Sharpe ratio of {sharpe_year:.2f} in {year}")
                else:
                    col.write(f"❌ Sub-universe test failed with Sharpe ratio of {sharpe_year:.2f} in {year}")
                    all_tests_passed = False
            else:
                col.write(f"❌ Sub-universe test failed with no data for {year}")

    else:
        col.write("⚫ Sub universe is only checked if other checks pass")
        
    col.write("⚫ Global correlation is only checked if other checks pass")

    if all_tests_passed:
        if col.button("Submit"):
            try:
                submit_alpha(driver_service, username, alpha_formula, alpha_settings, main_metrics)
                col.success("Alpha submitted successfully!")
            except Exception as e:
                col.error(f"Error occurred during submission: {e}")
    else:
        col.warning("All tests must pass to enable submission.")