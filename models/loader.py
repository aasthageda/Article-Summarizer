import os
import traceback
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = BASE_DIR / "training" / "lora_adapter_v2"

adapter_path = Path(
    os.environ.get("SUMMARIZER_ADAPTER_PATH", DEFAULT_ADAPTER)
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if adapter_path.is_dir():
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path)
        )
    else:
        model = base_model

    model.eval()
except Exception:
    traceback.print_exc()
    raise