#!/bin/bash
set -e

echo "Stopping any running containers..."
docker-compose down

echo "Rebuilding the Docker image..."
docker-compose build --no-cache

echo "Starting the container..."
docker-compose up -d

echo "Container status:"
docker-compose ps

echo "Container logs (press Ctrl+C to exit):"
docker-compose logs -f 