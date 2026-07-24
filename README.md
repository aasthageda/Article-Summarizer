# 📰 AI Article Summarizer

An end-to-end LLM-powered Article Summarizer built using Hugging Face Transformers, PEFT (LoRA), and Gradio.

The application takes a long article as input and generates a concise summary using a fine-tuned language model.

---

## Features

- Summarize long articles
- Fine-tuned using LoRA
- Gradio Web Interface
- Fast inference
- Hugging Face Transformers
- Easy deployment with Hugging Face Spaces

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Gradio

---

## Project Structure

```text
LLM_Project/
│
├── app.py
├── frontend/
│   └── ui.py
├── backend/
│   └── inference.py
├── models/
│   └── loader.py
├── training/
│   ├── train_lora.py
│   └── lora_adapter.py
├── requirements.txt
└── README.md
```

---

## How It Works

1. User enters an article.
2. Gradio sends the text to `inference.py`.
3. `loader.py` loads the tokenizer and model (only once).
4. The model generates a summary.
5. The summary is displayed back to the user.

---

## Installation

```bash
git clone https://github.com/yourusername/LLM_Project.git

cd LLM_Project

pip install -r requirements.txt

python app.py
```

---

## Screenshots

<img width="1920" height="1080" alt="Screenshot 2026-07-24 092206" src="https://github.com/user-attachments/assets/198f828d-e9e4-46f4-9083-4f874e5a759c" />
<img width="1920" height="1080" alt="Screenshot 2026-07-24 091815" src="https://github.com/user-attachments/assets/3d55406c-da7f-47ed-922a-3cd5321f4cd0" />


---

## Live Demo

Coming Soon

---

## Future Improvements

- Better summarization quality
- Beam Search optimization
- Support PDF summarization
- Multi-language summarization
- Docker deployment

---

## Author

Aastha Geda
MCA Student | AI & Machine Learning Enthusiast
