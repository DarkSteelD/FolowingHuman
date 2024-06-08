#!/bin/bash

# Navigate to your repository
cd /home/dark/Documents/GitHub/FolowingHuman || { echo "Failed to navigate to the repository"; exit 1; }

# Check if there are any changes
if [ -n "$(git status --porcelain)" ]; then
    # Add all changes
    git add .

    # Commit changes with a message
    git commit -m "Auto-commit: $(date)"

    # Push changes to the repository
    git push origin main # Change 'main' to your branch name if different
else
    echo "No changes to commit."
fi
