#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for LLM Data Processing Pipeline
"""

from pathlib import Path

# Global configuration
PROJECT_ID = "empathai-494308"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"
DEBUG_MODE = False  # Set to False for production run

# Paths - use absolute paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DPO_TRAIN_FILE = DATA_DIR / "dpo_train.jsonl"
DPO_VAL_FILE = DATA_DIR / "dpo_val.jsonl"
DPO_CLEANED_FILE = DATA_DIR / "dpo_cleaned.jsonl"
DPO_FINAL_FILE = DATA_DIR / "dpo_final.jsonl"
SFT_TRAIN_FILE = DATA_DIR / "sft_train.jsonl"
SFT_DEV_FILE = DATA_DIR / "sft_dev.jsonl"
SFT_TEST_FILE = DATA_DIR / "sft_test.jsonl"
DPO_TRAIN_SPLIT_FILE = DATA_DIR / "dpo_train_split.jsonl"
DPO_DEV_SPLIT_FILE   = DATA_DIR / "dpo_dev_split.jsonl"
DPO_TEST_SPLIT_FILE  = DATA_DIR / "dpo_test_split.jsonl"
FAILED_RECORDS_FILE  = DATA_DIR / "failed_records.jsonl"