from flask import Flask, request, render_template
from transformers import pipeline
import yake
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pymongo
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://UtkarshaSemantic9:UtkarshaSemantic9@customer-feedbacks.172va.mongodb.net/?retryWrites=true&w=majority&appName=Customer-Feedbacks")
db = client["sentiment_analysis_db"]
reviews_collection = db["reviews"]

# Load sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@app.route('/', methods=['GET', 'POST'])
def index():
    positive_aspects = []
    negative_aspects = []
    overall_result = {}
    user_text = ""

    if request.method == 'POST':
        user_text = request.form['review']

        # Overall Sentiment
        overall_result = sentiment_pipeline(user_text)[0]

        # Sentence-level analysis
        sentences = sent_tokenize(user_text)
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
        keywords = [kw for kw, _ in kw_extractor.extract_keywords(user_text)]

        for sentence in sentences:
            for aspect in keywords:
                if aspect.lower() in sentence.lower():
                    # Sentiment of aspect within its sentence
                    aspect_result = sentiment_pipeline(aspect + " " + sentence)[0]
                    sentiment = aspect_result['label']

                    if "1 star" in sentiment or "2 stars" in sentiment:
                        negative_aspects.append(aspect)
                    elif "4 stars" in sentiment or "5 stars" in sentiment:
                        positive_aspects.append(aspect)
                    # You can add "3 stars" as neutral if needed

        positive_aspects = list(set(positive_aspects))
        negative_aspects = list(set(negative_aspects))

        # MongoDB insertion
        review_data = {
            "review": user_text,
            "overall_sentiment": overall_result['label'],
            "confidence": round(overall_result['score'], 2),
            "positive_aspects": positive_aspects,
            "negative_aspects": negative_aspects
        }

        reviews_collection.insert_one(review_data)

    return render_template("index.html",
                           overall=overall_result,
                           positives=positive_aspects,
                           negatives=negative_aspects,
                           user_text=user_text)

if __name__ == '__main__':
    app.run(debug=True)
