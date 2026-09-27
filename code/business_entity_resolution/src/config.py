"""Central paths and hyper-parameters for the entity resolution pipeline."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.environ.get("ER_DATA_DIR", os.path.join(ROOT, "dataset"))
CACHE_DIR = os.environ.get("ER_CACHE_DIR", os.path.join(ROOT, "pipeline_cache"))
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OUTPUT_DIR = os.path.join(ROOT, "output")
EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")

SEED = 42
TOP_K = 100               # candidates kept per Source 1 record after blocking (val recall 0.933 IN / 0.952 US)
S1_CHUNK = 30_000         # Source 1 records processed per blocking chunk
CAND_P1_MIN = 0.001       # stage-1 LightGBM acts as a learned candidate filter: pairs with p1 below this are pruned
                          # before the stage-2 matcher (val: 91.7 -> 5.4 candidates / S1, 99.97 % of true links kept)
