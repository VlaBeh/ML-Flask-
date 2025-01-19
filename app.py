from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Завантажте модель
model = load_model('model_cnn_final.keras')

# Категорії розпізнавання
categories = ['Rock', 'Paper', 'Scissors']


# Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')


# Завантаження та обробка фото
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Перевірка та створення директорії 'uploads'
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Безпечне ім'я файлу
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    # Попередня обробка зображення
    image = load_img(filepath, target_size=(150, 150))  # Змінити розмір відповідно до вашої моделі
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Прогноз
    prediction = model.predict(image)
    result = categories[np.argmax(prediction)]

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
