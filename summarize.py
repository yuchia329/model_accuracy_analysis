# Use a pipeline as a high-level helper
from transformers import pipeline
from rouge_score import rouge_scorer
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"
pipe = pipeline("summarization", model="google/pegasus-xsum")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# 2) Provide the text you want to summarize
text = '''Reinforcement learning (RL) is a technique that trains software to make decisions to achieve the most optimal results. It mimics the trial-and-error learning process that humans use to achieve their goals. Software actions that work towards your goal are reinforced, while actions that detract from the goal are ignored.'''
# text = '''Once upon a time, in a small coastal village, a young girl named Mira woke to the sea’s gentle murmur each morning. She spent her days gathering shells along the shore and weaving stories into them, believing each shell held a tale from the ocean depths. As dusk settled, Mira would run to the edge of the water, tossing her shells back to the sea, letting them carry her soft-spoken wishes for grand adventures and magical encounters out into the world. Each night, she drifted off to sleep imagining the waves transforming her dreams into something real, waiting for her at dawn’s first light.
# '''

# text = '''"Oh great! I missed another deadline for my conference paper." Nilay forced a twisted grin and muttered, "I love my TA job." The teaching assistant job is taking up Nilay's sweet ass research time and requires him to throw all in to mentor these "genius" UCSC NLP students. The students know how to GPT answer their homework and have got tons of potential in the future. Thinking of it, Nilay chuckled and said, "I love being TA."'''

# 3) Tokenize the input text (convert it to model-ready format)
inputs = tokenizer(
    text,
    truncation=True,
    padding="longest",
    return_tensors="pt"  # PyTorch tensors
)

# 4) Generate the summary
summary_ids = model.generate(
    **inputs,
    max_length=60,
    min_length=10,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

# 5) Decode the generated summary back into text
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# 6) Compare with a reference using ROUGE
reference_summary = '''Reinforcement learning (RL) is a machine learning (ML) technique that trains software to make decisions to achieve the most optimal results.'''
# reference_summary = '''A young girl Mira lives an innocent and carefree life at a costal town, hoping her dream will come true one day.'''
# reference_summary = '''Nilay thinks being TA is wasting his time.'''

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)

print("Original Text:\n", text)
print("Generated Summary:", generated_summary)
print("ROUGE Scores:", scores)