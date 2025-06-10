#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building site..."
hugo

echo "Deploying..."
cd public
git add .
git commit -m "Site update: $(date)"
git push
cd ..
echo "Done!"
