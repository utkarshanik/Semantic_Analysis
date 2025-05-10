from transformers import pipeline
import yake
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load sentiment pipeline (Hugging Face model)
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Input product review
text = "Works as expected. The build quality is decent, and it performs well for everyday use. Charging is fast and battery life is acceptable. Overall, a good value for the price"

# Step 1: Overall sentiment analysis
overall_result = sentiment_pipeline(text)[0]
print(f"\n🔍 Overall Sentiment: {overall_result['label']} (Confidence: {overall_result['score']:.2f})\n")

# Step 2: Aspect Extraction + Sentence Sentiment
sentences = sent_tokenize(text)

# Extract potential aspects using YAKE
kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
keywords = [kw for kw, _ in kw_extractor.extract_keywords(text)]

# Containers for aspect-based sentiment
positive_aspects = []
negative_aspects = []

for sentence in sentences:
    result = sentiment_pipeline(sentence)[0]
    sentiment = result['label']

    # Determine polarity
    if "1 star" in sentiment or "2 stars" in sentiment:
        polarity = "Negative"
    elif "4 stars" in sentiment or "5 stars" in sentiment:
        polarity = "Positive"
    else:
        continue  # skip neutral

    # Match keywords (aspects) in sentence
    for aspect in keywords:
        if aspect.lower() in sentence.lower():
            if polarity == "Positive":
                positive_aspects.append(aspect)
            elif polarity == "Negative":
                negative_aspects.append(aspect)

# Remove duplicates
positive_aspects = list(set(positive_aspects))
negative_aspects = list(set(negative_aspects))

# Display results
print("✔️ Positive Aspects:")
for aspect in positive_aspects:
    print(f"  + {aspect}")

print("\n❌ Negative Aspects:")
for aspect in negative_aspects:
    print(f"  - {aspect}")




# from transformers import pipeline

# # Load the sentiment analysis pipeline using the nlptown model
# sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# # Example text for sentiment
# text = "performance is the most awesome"

# # Run sentiment analysis
# result = sentiment_pipeline(text)

# # Print result
# print("Sentiment Result:", result)



