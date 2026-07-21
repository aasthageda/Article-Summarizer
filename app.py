import os

print("APP.PY STARTED")

print("Importing frontend.ui...")
from frontend.ui import demo
print("frontend.ui imported successfully.")

if __name__ == "__main__":
    print("Launching Gradio...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )