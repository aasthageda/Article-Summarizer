from models.loader import tokenizer, model


def summarize(article):
    if not article.strip():
        return "Please enter an article."
    text =  article
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )


    summary_id = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=160,
        min_new_tokens=25,
        num_beams=4,
        length_penalty=1.5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    summary = tokenizer.decode(
        summary_id[0],
        skip_special_tokens=True
    )

    #print(summary)
    return summary
def clear():
    return "",""
