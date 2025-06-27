from flask import Flask, render_template, request
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model("model_cnn (2).h5")  # Make sure this file is in your project folder

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# List your model's output classes here
classes = ['floral', 'geometric', 'plaid', 'polka dot', 'striped']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))  # Use your model’s expected input size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('predictionpage.html', prediction=predicted_class, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
