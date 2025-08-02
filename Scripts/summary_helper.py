from transformers import pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return (summary[0]['summary_text'])