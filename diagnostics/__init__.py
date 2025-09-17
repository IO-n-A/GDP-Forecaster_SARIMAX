"""Diagnostic testing and robust inference for GDP-ForecasterSARIMAX.

This package provides comprehensive diagnostic testing and robust statistical
inference capabilities including:
- Heteroskedasticity detection (ARCH-LM, Breusch-Pagan tests)
- Robust standard error computation (HAC/Newey-West)
- Residual diagnostics and model validation
- GARCH modeling for heteroskedastic residuals
- Integration with SARIMAX models
"""

from .heteroskedasticity import (
    HeteroskedasticityTester,
    HeteroskedasticityResult,
    RobustInferenceManager,
    test_heteroskedasticity,
    apply_robust_inference
)

from .residual_diagnostics import (
    ResidualDiagnostics,
    DiagnosticResult,
    DiagnosticTest,
    run_comprehensive_diagnostics
)

__all__ = [
    # Heteroskedasticity testing
    'HeteroskedasticityTester',
    'HeteroskedasticityResult', 
    'RobustInferenceManager',
    'test_heteroskedasticity',
    'apply_robust_inference',
    
    # Residual diagnostics
    'ResidualDiagnostics',
    'DiagnosticResult',
    'DiagnosticTest',
    'run_comprehensive_diagnostics'
]

# Version info
__version__ = '1.0.0'