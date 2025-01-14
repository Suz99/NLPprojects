from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Function to summarize text
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=50, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Define API route for summarization
@app.route("/summarize", methods=["POST"])
def get_summary():
    data = request.get_json()
    article = data.get("article", "")
    if not article:
        return jsonify({"error": "No article provided"}), 400

    summary = summarize(article)
    return jsonify({"summary": summary})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
