#!/bin/bash

# Activate your mamba environment
mamba activate tcc

# Loop indefinitely
while true; do
   # Run your Python script
   python traintest.py

   # Optional: sleep for a specified time (e.g., 1 second)
   sleep 5
done