#submissions.py
import pandas as pd
import numpy as np
import streamlit as st
from utils.alpha_db import submit_alpha, load_all_submitted_alphas

def run_tests(metrics):
    test_results = {
        "Sharpe": (metrics["Sharpe"], 1),
        "Fitness": (metrics["Fitness"], 1),
        "Turnover (%)": (metrics["Turnover (%)"], (0, 70)),
        "Weight Concentration": ("Passed", None),
    }
    
    return test_results

def check_correlation(drive_service, username, current_alpha_weights):
    alpha_df = load_all_submitted_alphas(drive_service)
    correlation_threshold = 0.6

    if alpha_df.empty:
        return False, 0
    
    for _, submitted_alpha in alpha_df.iterrows():
        previous_alpha_weights = np.fromstring(submitted_alpha["Weight Metrics"][1:-1], sep=' ')
        previous_alpha_weights = previous_alpha_weights.reshape(current_alpha_weights.shape)

        correlation = np.corrcoef(current_alpha_weights.flatten(), previous_alpha_weights.flatten())[0, 1]

        if correlation > correlation_threshold:
            return True, correlation
    return False, None

def show_test_results(test_results, col, alpha_formula, alpha_settings, main_metrics, sub_universe_metrics, result, driver_service, username):
    col.subheader("Test Results")

    all_tests_passed = True
    weight_passed = turnover_passed = fitness_passed = sharpe_passed = False
    sub_universe_passed = False
    main_test_passed = False
    correlation_test_passed = False

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
        main_test_passed = True
    else:
        all_tests_passed = False
    
    if main_test_passed:
        sub_universe_passed = True
        for year in [2022, 2023]:
            if year in sub_universe_metrics.index:
                sharpe_year = sub_universe_metrics.loc[year, "Sharpe"]
                if sharpe_year >= 1.0:
                    col.write(f"✅ Sub-universe test passed with Sharpe ratio of {sharpe_year:.2f} in {year}")
                else:
                    col.write(f"❌ Sub-universe test failed with Sharpe ratio of {sharpe_year:.2f} in {year}")
                    sub_universe_passed = False
                    all_tests_passed = False
            else:
                col.write(f"❌ Sub-universe test failed with no data for {year}")
                sub_universe_passed = False
                all_tests_passed = False
    else:
        col.write("⚫ Sub-universe test is only checked if main tests pass.")
    
    if sub_universe_passed:
        correlation_failed, correlation_value = check_correlation(driver_service, username, result)
        if correlation_failed:
            col.write(f"❌ Global correlation test failed with correlation of {correlation_value:.2f}, above the cut-off of 0.6")
            all_tests_passed = False
        else:
            col.write(f"✅ Global correlation test passed with correlation below the cut-off of 0.6")
            correlation_test_passed = True
    else:
        col.write("⚫ Correlation test is only checked if other tests pass.")
    
    if correlation_test_passed:
        all_tests_passed = True
    else:
        all_tests_passed = False

    if all_tests_passed:
        if col.button("Submit"):
            try:
                submit_alpha(driver_service, username, alpha_formula, alpha_settings, main_metrics)
                col.success("Alpha submitted successfully!")
            except Exception as e:
                col.error(f"Error occurred during submission: {e}")
    else:
        col.warning("All tests must pass to enable submission.")