#!/usr/bin/env python3
"""
Command-line interface for NukeSamurai API.
"""
import os
import sys
import argparse
import subprocess
import uvicorn
from pathlib import Path
from loguru import logger

def run_server():
    """
    Run the NukeSamurai API server with optional arguments.
    """
    parser = argparse.ArgumentParser(description="Run the NukeSamurai API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--log-level", default="info", help="Log level (debug, info, warning, error, critical)")
    args = parser.parse_args()

    logger.info(f"Starting NukeSamurai API server on {args.host}:{args.port}")
    logger.info(f"Log level: {args.log_level}")
    
    # Check if model checkpoints exist
    checkpoints_dir = Path("NukeSamurai/sam2_repo/sam2/checkpoints")
    if not checkpoints_dir.exists() or not any(checkpoints_dir.glob("*.pt")):
        logger.warning("SAM2 model checkpoints not found. You may need to download them.")
        logger.warning("Run: cd NukeSamurai/sam2_repo/sam2/checkpoints && ./download_checkpoints.sh")
    
    # Run the server
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

def run_tests():
    """
    Run the test suite for NukeSamurai API.
    """
    parser = argparse.ArgumentParser(description="Run NukeSamurai API tests")
    parser.add_argument("--api-tests", action="store_true", help="Run API endpoint tests using tester.sh")
    parser.add_argument("--unit-tests", action="store_true", help="Run unit tests using pytest")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Default to all tests if no specific test type is specified
    if not (args.api_tests or args.unit_tests):
        args.all = True

    # Run API tests if requested
    if args.api_tests or args.all:
        logger.info("Running API endpoint tests...")
        # Check if tester.sh exists
        if not os.path.exists("tester.sh"):
            logger.error("tester.sh script not found. Cannot run API tests.")
        else:
            # Create output directory if it doesn't exist
            os.makedirs("Output", exist_ok=True)
            
            # Run the tester.sh script
            try:
                subprocess.run(["bash", "tester.sh"], check=True)
                logger.info("API tests completed successfully.")
            except subprocess.CalledProcessError:
                logger.error("API tests failed.")
                sys.exit(1)

    # Run unit tests if requested
    if args.unit_tests or args.all:
        logger.info("Running unit tests...")
        try:
            subprocess.run(["pytest", "-xvs", "tests"], check=True)
            logger.info("Unit tests completed successfully.")
        except subprocess.CalledProcessError:
            logger.error("Unit tests failed.")
            sys.exit(1)

if __name__ == "__main__":
    run_server() 