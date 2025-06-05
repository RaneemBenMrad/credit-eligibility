import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Charger le modèle depuis le fichier pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Récupérer les données saisies par l'utilisateur
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Effectuer la prédiction avec le modèle
    prediction = model.predict(final_features)
    output = prediction[0]  # Supposons que 1 = éligible, 0 = non éligible

    # Logique conditionnelle pour afficher un message adapté
    if output == 1:
        prediction_text = 'Félicitations ! Vous êtes éligible pour obtenir un crédit bancaire.'
    else:
        prediction_text = 'Désolé, vous ne semblez pas répondre aux critères pour un crédit bancaire.'

    # Retourner le résultat à la page index.html
    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
