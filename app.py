import os
import json
import requests
from flask import Flask, request, Response
import joblib
import tensorflow as tf

app = Flask(__name__)

# Загрузка модели и скалера
model_path = 'model/lstm_model.keras'
scaler_path = 'model/scaler.save'
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Функция для получения данных с CoinGecko
def get_coingecko_data(token):
    url = f"https://api.coingecko.com/api/v3/coins/{token}/market_chart?vs_currency=usd&days=30&interval=daily"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "твой_ключ_CoinGecko"  # замени на свой ключ
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.status_code} {response.text}")
    
    return response.json()

# Основной роутинг для обработки запросов
@app.route("/inference/<string:token>", methods=["POST"])
def get_inference(token):
    try:
        # Получаем данные с CoinGecko
        coingecko_data = get_coingecko_data(token)
        
        # Предположим, что данные цены находятся в 'prices'
        prices = [item['close'] for item in coingecko_data['prices']]  # корректируй в зависимости от структуры данных
        
        # Подготовка данных для модели
        scaled_data = scaler.transform([prices])
        scaled_data = scaled_data.reshape((1, len(prices), 1))
        
        # Прогнозирование
        prediction = model.predict(scaled_data)
        predicted_price = scaler.inverse_transform(prediction)[0][0]
        
        return Response(json.dumps({"predicted_price": predicted_price}), status=200, mimetype='application/json')
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
