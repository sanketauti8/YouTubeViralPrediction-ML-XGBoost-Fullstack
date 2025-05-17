from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("youtube_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    title = data.get('title', '')
    description = data.get('description', '')
    tags = data.get('tags', '[none]')

    title_length = len(title)
    desc_length = len(description)
    num_tags = 0 if tags == '[none]' else len(tags.split('|'))
    all_caps_title = int(title.isupper())

    input_data = pd.DataFrame([{
        'likes': data['likes'],
        'dislikes': data['dislikes'],
        'comment_count': data['comment_count'],
        'title_length': title_length,
        'desc_length': desc_length,
        'num_tags': num_tags,
        'all_caps_title': all_caps_title,
        'publish_hour': data['publish_hour'],
        'publish_day': data['publish_day']
    }])

    prediction = model.predict(input_data)[0]
    return jsonify({"viral": bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
