#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [animals|fruits|both]"
    exit 1
fi

# Loop indefinitely
while true; do
    # Activate your mamba environment
   #  mamba activate tcc

    case $1 in
        animals)
            # Run your Python script with 'animals' argument
            python3 traintest.py animals
            ;;

        fruits)
            # Run your Python script with 'fruits' argument
            python3 traintest.py fruits
            ;;

        both)
            # Alternate between 'animals' and 'fruits'
            python3 traintest.py animals
            sleep 5 # Sleep between commands
            python3 traintest.py fruits
            ;;

        *)
            echo "Invalid argument. Please use 'animals', 'fruits', or 'both'."
            exit 2
            ;;
    esac

    # Optional: sleep for a specified time (e.g., 1 second)
    sleep 5

    # Deactivate the environment (optional, if needed)
    # conda deactivate
done