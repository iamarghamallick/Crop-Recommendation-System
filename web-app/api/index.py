import os
from dotenv import dotenv_values
import json
import random
from flask import Flask, request, render_template
import numpy as np
import pickle
import requests

config = dotenv_values(".env")

# importing model
model = pickle.load(open('exported_model/model.pkl','rb'))
sc = pickle.load(open('exported_model/standscaler.pkl','rb'))
ms = pickle.load(open('exported_model/minmaxscaler.pkl','rb'))

# random image generation of crop
def generate_image(crop):
    CLIENT_ID = os.getenv("CLIENT_ID")
    url = f"https://api.unsplash.com/search/photos?query={crop}&client_id={CLIENT_ID}"
    img_url = "https://upload.wikimedia.org/wikipedia/commons/b/b4/Manuring_a_vegetable_garden.jpg"  # default
    try:
        response = requests.get(url)
        response_text = json.loads(response.text)
        if response_text['results']:
            results = response_text['results']
            img_url = results[random.randint(
                0, len(results)-1)]['urls']['small']
    except:
        print(f"Error generating the requested image: {crop}")
    return img_url

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    print(prediction[0])

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "According to our AI Model, {} is the best crop to be cultivated right there based on your given data.".format(crop)
        input_data = f"""
 Nitrogen: {N}%, Phosporus: {P}%
 Potassium: {K}%, Temperature: {temp} °C
 Humidity: {humidity}, pH: {ph}
 Rainfall: {rainfall} mm"""
        img_url = generate_image(crop)
        print(input_data)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result, crop = crop, input_data = input_data, img_url=img_url)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")