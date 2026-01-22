#!/usr/bin/env python3
"""
Run the scheduler worker for automated data sync, training, and predictions.

Usage:
    python run_scheduler.py

This script will:
1. Sync data from Yahoo Finance every 30 minutes
2. Train ML models daily at 6 AM IST
3. Generate predictions every 30 minutes
4. Verify predictions every 5 minutes
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from workers.scheduler import run_scheduler

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("scheduler.log"),
        ],
    )

    print("Starting Silver Prediction Scheduler...")
    print("Press Ctrl+C to stop")

    asyncio.run(run_scheduler())
