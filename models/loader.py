import os
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/bart-large-cnn"
BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_ADAPTER = BASE_DIR / "training" / "lora_adapter_v2"
adapter_path = Path(os.environ.get("SUMMARIZER_ADAPTER_PATH", DEFAULT_ADAPTER))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, str(adapter_path)) if adapter_path.is_dir() else base_model
model.eval()
