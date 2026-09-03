@echo off
setlocal enabledelayedexpansion

rem Runs the model training pipeline, then verifies the expected
rem pkl/joblib artifacts were actually produced.
rem
rem Usage (from anywhere): run_training.bat

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "ARTIFACTS_DIR=%PROJECT_ROOT%artifacts"
set "MODELS_DIR=%ARTIFACTS_DIR%\models"

echo [1/2] Running training pipeline...
python -m src.models.train
if errorlevel 1 (
    echo ERROR: training pipeline exited with an error. 1>&2
    exit /b 1
)

echo.
echo [2/2] Verifying artifacts...
set "missing=0"

call :check_file "%ARTIFACTS_DIR%\best_model.pkl"
call :check_file "%ARTIFACTS_DIR%\best_model.joblib"
call :check_file "%ARTIFACTS_DIR%\eval_bundle.joblib"
call :check_file "%ARTIFACTS_DIR%\model_comparison.csv"
call :check_file "%ARTIFACTS_DIR%\test_predictions.csv"
call :check_file "%MODELS_DIR%\logistic_regression.joblib"
call :check_file "%MODELS_DIR%\lightgbm.joblib"
call :check_file "%MODELS_DIR%\xgboost.joblib"
call :check_file "%MODELS_DIR%\svm.joblib"

echo.
if "%missing%"=="1" (
    echo FAILED: one or more expected pkl/joblib artifacts were not created. 1>&2
    exit /b 1
)

echo SUCCESS: all model artifacts present.
exit /b 0

:check_file
if exist "%~1" (
    for %%F in ("%~1") do echo   OK   ^(%%~zF bytes^)  %~1
) else (
    echo   MISSING        %~1
    set "missing=1"
)
exit /b 0
