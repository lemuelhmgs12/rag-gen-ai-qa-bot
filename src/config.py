"""
project settings - all notebooks use the same values.
"""
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "abc-grocery-help-desk-data.md"

# Where Chroma saves the vector database
VECTOR_DB_DIR = PROJECT_ROOT / "abc_vector_db_chroma"

# model
LLM_MODEL = "gpt-5"
EMBEDDING_MODEL = "text-embedding-3-small"

# vector db and retrieval
COLLECTION_NAME = "abc_help_qa"
K = 4
SCORE_THRESHOLD = 0.33