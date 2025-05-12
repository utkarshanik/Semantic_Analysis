from fastapi import FastAPI
from pymongo import MongoClient
from bson.json_util import dumps
from fastapi.responses import JSONResponse
from fastapi import Response

app = FastAPI()

MONGO_URI = "mongodb+srv://UtkarshaSemantic9:UtkarshaSemantic9@customer-feedbacks.172va.mongodb.net/?retryWrites=true&w=majority&appName=Customer-Feedbacks"

client = MongoClient(MONGO_URI)
db = client["sentiment_analysis_db"]
collection = db["reviews"]

@app.get("/data")
async def get_data():
    reviews = list(collection.find())
    return Response(content=dumps(reviews), media_type="application/json")
