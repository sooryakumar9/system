from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = """
The Hugging Face Transformers library provides a simple and consistent API for using a wide variety of pre-trained models 
for natural language processing tasks such as text classification, question answering, and summarization. 
One of its most powerful features is the pipeline abstraction, which allows users to perform these tasks 
with just a few lines of code. Summarization is the task of shortening a long document while preserving 
its key information and overall meaning. Huggingx Face offers several models fine-tuned for summarization, 
such as BART and T5. These models can be accessed through the summarization pipeline, and they can generate 
high-quality summaries that are useful for a variety of applications, such as news aggregation, document review, 
and information retrieval.
"""

summary = summarizer(long_text, max_length=100, min_length=60, do_sample=False)

print("Summary:", summary[0]['summary_text'])

"""
if summary and isinstance(summary, list):
    print("Summary:", summary[0]['summary_text'])
else:
    print("Unexpected output:", summary)
"""