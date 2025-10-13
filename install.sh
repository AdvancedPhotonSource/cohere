#!/bin/bash

pip install cohere_core
pip install cohere_ui
pip install cohere_beamlines

if [ "$#" -ne 0 ]; then
  if [ "$1" = "cupy" ]; then
    conda install cupy=12.2.0 -c conda-forge
  fi
  if [ "$1" = "torch" ]; then
    pip install torch
  fi
fi
