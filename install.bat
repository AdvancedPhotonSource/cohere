@echo off

:: Install Python packages
pip install cohere_core
pip install cohere_ui
pip install cohere_beamlines

:: Check for arguments
IF NOT "%~1"=="" (
    IF "%~1"=="cupy" (
        :: Install cupy via conda
        conda install cupy=12.2.0 -c conda-forge
    )
    IF "%~1"=="torch" (
        :: Install torch via pip
        pip install torch
    )
)
