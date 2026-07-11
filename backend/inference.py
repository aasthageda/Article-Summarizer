from models.loader import tokenizer, model
def summarize(text):
    if not text.strip():
        return "Please enter an article."
    inputs = tokenizer(
        "summarize: " + text,
        #text,
        return_tensors="pt"
    )
    summary_id = model.generate(**inputs)
    summary = tokenizer.decode(
        summary_id[0],
        skip_special_tokens=True
    )

    #print(summary)
    return summary
def clear():
    return "",""
