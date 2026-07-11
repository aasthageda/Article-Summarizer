import gradio as gr
from backend.inference import summarize,clear

with gr.Blocks(title="Article Summarizer",theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Article Summarizer")

    gr.Markdown(
        "Summarize long articles using **DistilBART (google/flan-t5)**"
    )

    with gr.Row():
        title = " Article Summarizer"
        article = gr.Textbox(
            label="Article",
            lines=25,
            placeholder="Paste your article here..."
        )

        summary = gr.Textbox(
            label="Summary",
            lines=25
        )

    btn = gr.Button("Summarize")

    btn.click(
        fn=summarize,
        inputs=article,
        outputs=summary
    )
    clear_btn = gr.Button("Clear All")
    clear_btn.click(
        fn=clear,
        outputs=[article, summary]
    )

