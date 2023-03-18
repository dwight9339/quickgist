from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, num_return_sequences=1)
    return [tokenizer.decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for summary_id in summary_ids]

@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    text = data["text"]
    summary = generate_summary(text)
    return jsonify(summary[0])

if __name__ == "__main__":
    app.run(debug=True)