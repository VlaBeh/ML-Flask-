from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

model = load_model('model_cnn_final.keras')


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels = ['rock', 'paper', 'scissor']


def predict_image(image_path):
    # Завантажуємо зображення
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)  # Перетворюємо в масив
    img = img / 255.0  # Нормалізація
    img = np.expand_dims(img, axis=0)  # Додаємо розмірність для батчу

    # Отримуємо передбачення
    prediction = model.predict(img)
    predicted_class = labels[np.argmax(prediction)]  # Отримуємо клас з найбільшою ймовірністю
    return predicted_class


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # Зберігаємо зображення
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Передбачення
        prediction = predict_image(file_path)
        return render_template('index.html', prediction=prediction, img_path=file_path)


if __name__ == '__main__':
    app.run()
