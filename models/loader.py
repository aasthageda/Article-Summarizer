from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_NAME = "google/flan-t5-small"
BASE_DIR = Path(__file__).resolve().parents[1]
ADAPTER_PATH = str(BASE_DIR / "training" / "lora_adapter")



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)
model.eval()