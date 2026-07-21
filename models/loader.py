import os
import traceback
from pathlib import Path

print("1. loader.py imported")

from peft import PeftModel
print("2. peft imported")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print("3. transformers imported")

MODEL_NAME = "facebook/bart-large-cnn"

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = BASE_DIR / "training" / "lora_adapter_v2"

adapter_path = Path(
    os.environ.get("SUMMARIZER_ADAPTER_PATH", DEFAULT_ADAPTER)
)

try:
    print("4. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("5. Loading base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("6. Adapter exists:", adapter_path.exists())

    if adapter_path.is_dir():
        print("7. Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        print("7. Using base model")
        model = base_model

    print("8. Model ready")
    model.eval()

except Exception:
    traceback.print_exc()
    raise