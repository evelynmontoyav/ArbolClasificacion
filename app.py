from flask import Flask, render_template, request
import joblib
import os
import numpy as np

# Cargar el modelo entrenado
model_path = os.path.join(os.path.dirname(__file__), 'models', 'regresion_arbol.pkl')
model = joblib.load(model_path)

# Crear una aplicación Flask
app = Flask(__name__)

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'

# Definir la ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperatura = int(request.form['temperatura'])
    humedad = int(request.form['humedad'])
    ph = int(request.form['ph'])
    precipitacion = int(request.form['precipitacion'])
    
    # Realizar una predicción de probabilidades utilizando el modelo cargado
    pred_probabilities = np.array([[N, P, K, temperatura, humedad, ph, precipitacion]])
    
    # Obtener los nombres de las clases (Deserción, Alerta, Buen estudiante)
    prediccion = model.predict(pred_probabilities)

    # Renderizar la plantilla 'result.html' y pasar el mensaje a la plantilla
    return render_template('result.html', pred=prediccion[0])

# Iniciar la aplicación si este script es el punto de entrada
if __name__ == '__main__':
    app.run(debug=True)
