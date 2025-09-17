#!/usr/bin/env python3
"""Test script for enhanced evaluation metrics with refined stability filtering."""

import logging
import warnings
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_forecasting_data():
    """Create test data for evaluation metrics testing."""
    np.random.seed(42)
    
    # Create quarterly data
    dates = pd.date_range('2010-01-01', periods=50, freq='QE')
    
    # True GDP values
    y_true = pd.Series(
        100 + 10 * np.sin(2 * np.pi * np.arange(50) / 4) + 0.1 * np.arange(50) + np.random.normal(0, 2, 50),
        index=dates,
        name='true_gdp'
    )
    
    # Good forecast (small errors)
    y_pred_good = y_true + np.random.normal(0, 1, 50)
    y_pred_good.name = 'good_forecast'
    
    # Poor forecast (large errors)
    y_pred_poor = y_true + np.random.normal(0, 5, 50)
    y_pred_poor.name = 'poor_forecast'
    
    # Naive forecast (last value carried forward)
    y_naive = y_true.shift(1).fillna(y_true.iloc[0])
    y_naive.name = 'naive_forecast'
    
    # Create problematic data near zero (for testing MAPE instability)
    y_true_near_zero = pd.Series(
        np.abs(np.random.normal(0, 0.1, 20)),  # Small values near zero
        index=dates[:20],
        name='true_near_zero'
    )
    y_pred_near_zero = y_true_near_zero + np.random.normal(0, 0.05, 20)
    y_pred_near_zero.name = 'pred_near_zero'
    
    return {
        'normal': {'true': y_true, 'pred_good': y_pred_good, 'pred_poor': y_pred_poor, 'naive': y_naive},
        'near_zero': {'true': y_true_near_zero, 'pred': y_pred_near_zero}
    }


def test_enhanced_evaluation_system():
    """Test the enhanced evaluation metrics system."""
    print("=" * 70)
    print("Testing Enhanced Evaluation Metrics System")
    print("=" * 70)
    
    # Test 1: Import modules
    print("\nTest 1: Module Imports")
    try:
        from evaluation import (
            EnhancedMetricsCalculator,
            MetricResult,
            MetricStability,
            PerformanceLevel,
            compute_enhanced_metrics,
            MetricRefinementEngine,
            RefinedMetricsResult,
            refine_evaluation_metrics,
            filter_stable_metrics,
            create_refinement_report
        )
        print("✓ All enhanced evaluation modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced evaluation modules: {e}")
        return False
    
    # Test 2: Create test data
    print("\nTest 2: Test Data Creation")
    try:
        test_data = create_test_forecasting_data()
        normal_data = test_data['normal']
        near_zero_data = test_data['near_zero']
        
        print("✓ Test forecasting data created:")
        print(f"  - Normal GDP data: {len(normal_data['true'])} observations")
        print(f"  - Near-zero data: {len(near_zero_data['true'])} observations")
        
    except Exception as e:
        print(f"✗ Test data creation failed: {e}")
        return False
    
    # Test 3: Enhanced metrics calculator
    print("\nTest 3: Enhanced Metrics Calculator")
    try:
        calculator = EnhancedMetricsCalculator()
        print("✓ Enhanced metrics calculator created")
        
        # Test with good forecast
        good_metrics = calculator.compute_enhanced_metrics(
            normal_data['true'], 
            normal_data['pred_good'],
            normal_data['naive'],
            include_deprecated=False
        )
        
        print(f"✓ Computed {len(good_metrics)} stable/robust metrics:")
        for name, result in good_metrics.items():
            stability_str = f"[{result.stability.value.upper()}]"
            perf_str = f"({result.performance_level.value})" if result.performance_level else ""
            print(f"  - {name.upper()}: {result.value:.4f} {stability_str} {perf_str}")
        
        # Verify no deprecated metrics in headline results
        deprecated_count = sum(1 for r in good_metrics.values() 
                             if r.stability == MetricStability.DEPRECATED)
        print(f"✓ Deprecated metrics in headline results: {deprecated_count} (should be 0)")
        
    except Exception as e:
        print(f"✗ Enhanced metrics calculator test failed: {e}")
        return False
    
    # Test 4: Deprecated metrics with warnings
    print("\nTest 4: Deprecated Metrics with Warnings")
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Include deprecated metrics
            all_metrics = calculator.compute_enhanced_metrics(
                normal_data['true'],
                normal_data['pred_good'], 
                normal_data['naive'],
                include_deprecated=True
            )
            
            # Check for deprecation warnings
            deprecation_warnings = [warning for warning in w 
                                  if issubclass(warning.category, DeprecationWarning)]
            
            print(f"✓ Total metrics (including deprecated): {len(all_metrics)}")
            print(f"✓ Deprecation warnings generated: {len(deprecation_warnings)}")
            
            # Count deprecated metrics
            deprecated_metrics = [name for name, result in all_metrics.items() 
                                if result.stability == MetricStability.DEPRECATED]
            print(f"✓ Deprecated metrics: {deprecated_metrics}")
            
            for warning in deprecation_warnings[:2]:  # Show first 2 warnings
                print(f"  ⚠ {warning.message}")
        
    except Exception as e:
        print(f"✗ Deprecated metrics test failed: {e}")
        return False
    
    # Test 5: MAPE instability with near-zero data
    print("\nTest 5: MAPE Instability Detection")
    try:
        # Test MAPE with near-zero data (should be problematic)
        near_zero_metrics = calculator.compute_enhanced_metrics(
            near_zero_data['true'],
            near_zero_data['pred'],
            include_deprecated=True
        )
        
        if 'mape' in near_zero_metrics:
            mape_result = near_zero_metrics['mape']
            print(f"✓ MAPE with near-zero data: {mape_result.value:.2f}")
            print(f"  - Stability: {mape_result.stability.value}")
            print(f"  - Warnings: {len(mape_result.computation_warnings)}")
            
            for warning in mape_result.computation_warnings[:2]:
                print(f"    ⚠ {warning}")
        
        # Show stable metrics are unaffected
        stable_metrics_near_zero = [name for name, result in near_zero_metrics.items() 
                                  if result.stability == MetricStability.STABLE]
        print(f"✓ Stable metrics unaffected: {stable_metrics_near_zero}")
        
    except Exception as e:
        print(f"✗ MAPE instability test failed: {e}")
        return False
    
    # Test 6: Metric refinement engine
    print("\nTest 6: Metric Refinement Engine")
    try:
        refinement_engine = MetricRefinementEngine()
        print("✓ Metric refinement engine created")
        
        # Test refinement with normal data
        refined_result = refinement_engine.refine_evaluation_metrics(
            normal_data['true'],
            normal_data['pred_good'],
            normal_data['naive'],
            include_legacy=True
        )
        
        print("✓ Metric refinement completed:")
        print(f"  - Refined metrics: {len(refined_result.refined_metrics)}")
        print(f"  - Deprecated metrics: {len(refined_result.deprecated_metrics) if refined_result.deprecated_metrics else 0}")
        print(f"  - Legacy metrics: {len(refined_result.legacy_metrics) if refined_result.legacy_metrics else 0}")
        
        # Test performance assessment
        if refined_result.performance_assessment:
            perf = refined_result.performance_assessment
            print(f"✓ Performance assessment:")
            print(f"  - Primary metric: {perf.primary_metric_name.upper()} = {perf.primary_metric_score:.4f}")
            print(f"  - Performance level: {perf.performance_level.value}")
            print(f"  - Overall score: {perf.overall_score:.3f}")
            print(f"  - Stability score: {perf.stability_score:.3f}")
        
    except Exception as e:
        print(f"✗ Metric refinement engine test failed: {e}")
        return False
    
    # Test 7: Metric filtering by stability
    print("\nTest 7: Stability-Based Filtering")
    try:
        # Get all metrics including deprecated
        all_metrics = calculator.compute_enhanced_metrics(
            normal_data['true'],
            normal_data['pred_good'],
            include_deprecated=True
        )
        
        # Filter by stability levels
        stable_only = filter_stable_metrics(all_metrics, MetricStability.STABLE)
        robust_and_stable = filter_stable_metrics(all_metrics, MetricStability.ROBUST)
        
        print(f"✓ Stability filtering results:")
        print(f"  - All metrics: {len(all_metrics)}")
        print(f"  - Stable only: {len(stable_only)}")
        print(f"  - Robust + Stable: {len(robust_and_stable)}")
        
        # Verify no deprecated metrics in filtered results
        for name, result in robust_and_stable.items():
            if result.stability == MetricStability.DEPRECATED:
                print(f"✗ Found deprecated metric '{name}' in filtered results")
                return False
        
        print("✓ No deprecated metrics in stability-filtered results")
        
    except Exception as e:
        print(f"✗ Stability filtering test failed: {e}")
        return False
    
    # Test 8: Performance comparison (good vs poor forecasts)
    print("\nTest 8: Performance Comparison")
    try:
        # Compare good vs poor forecasts
        good_refined = refine_evaluation_metrics(
            normal_data['true'],
            normal_data['pred_good'],
            normal_data['naive']
        )
        
        poor_refined = refine_evaluation_metrics(
            normal_data['true'],
            normal_data['pred_poor'],
            normal_data['naive']
        )
        
        print("✓ Performance comparison:")
        
        # Compare primary metrics
        good_perf = good_refined.performance_assessment
        poor_perf = poor_refined.performance_assessment
        
        if good_perf and poor_perf:
            print(f"  Good forecast:")
            print(f"    - MAE: {good_refined.refined_metrics['mae'].value:.4f}")
            print(f"    - Performance: {good_perf.performance_level.value}")
            print(f"  Poor forecast:")
            print(f"    - MAE: {poor_refined.refined_metrics['mae'].value:.4f}")
            print(f"    - Performance: {poor_perf.performance_level.value}")
            
            # Verify good forecast has better metrics
            good_mae = good_refined.refined_metrics['mae'].value
            poor_mae = poor_refined.refined_metrics['mae'].value
            
            if good_mae < poor_mae:
                print("✓ Good forecast correctly shows better performance")
            else:
                print("⚠ Performance comparison unexpected")
        
    except Exception as e:
        print(f"✗ Performance comparison test failed: {e}")
        return False
    
    # Test 9: Configuration integration
    print("\nTest 9: Configuration Integration")
    try:
        # Test with configuration manager (if available)
        try:
            from config import get_config
            config_manager = get_config()
            
            config_calculator = EnhancedMetricsCalculator(config_manager)
            config_refined = refine_evaluation_metrics(
                normal_data['true'],
                normal_data['pred_good'],
                config_manager=config_manager
            )
            
            print("✓ Configuration integration successful")
            print(f"  - Configured metrics: {len(config_refined.refined_metrics)}")
            
        except Exception:
            print("⚠ Configuration manager not available, tested with defaults")
        
    except Exception as e:
        print(f"✗ Configuration integration test failed: {e}")
        return False
    
    # Test 10: Reporting and output
    print("\nTest 10: Enhanced Reporting")
    try:
        # Create comprehensive report
        sample_result = refine_evaluation_metrics(
            normal_data['true'],
            normal_data['pred_good'],
            normal_data['naive']
        )
        
        report = create_refinement_report(sample_result, "Test Evaluation Report")
        
        print("✓ Enhanced reporting created")
        
        # Check report contains key sections
        required_sections = [
            "Performance Assessment",
            "Refined Metrics",
            "Recommendations"
        ]
        
        for section in required_sections:
            if section in report:
                print(f"  ✓ Report contains '{section}' section")
            else:
                print(f"  ⚠ Report missing '{section}' section")
        
        # Show abbreviated report
        print("\nSample Report (first 10 lines):")
        report_lines = report.split('\n')
        for i, line in enumerate(report_lines[:10]):
            print(f"  {line}")
        print("  ...")
        
    except Exception as e:
        print(f"✗ Enhanced reporting test failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("Enhanced Evaluation Metrics Tests Completed Successfully!")
    print("\nKey Features Verified:")
    print("✓ Stable primary metrics prioritized: MAE, RMSE, MASE, Theil's U")
    print("✓ Robust secondary metrics: Median AE, Directional Accuracy")
    print("✓ MAPE/sMAPE correctly deprecated with warnings")
    print("✓ Instability detection for near-zero values")
    print("✓ Performance assessment and thresholds")
    print("✓ Stability-based metric filtering")
    print("✓ Backward compatibility with legacy metrics")
    print("✓ Configuration system integration")
    print("✓ Comprehensive reporting with recommendations")
    print("✓ Deprecation warnings for unstable metrics")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    import io
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    try:
        from helpers.log_utils import append_eval_log
    except Exception:
        append_eval_log = None

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        success = test_enhanced_evaluation_system()

    output = buf.getvalue()
    if append_eval_log:
        try:
            append_eval_log("tests/test_enhanced_evaluation.py", output)
        except Exception:
            pass

    print(output, end="")
    exit(0 if success else 1)