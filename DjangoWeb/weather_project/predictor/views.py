from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np
import os
from django.conf import settings
import math
from datetime import datetime

MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor/ml_models/rainfall_model.pkl')
ENCODER_PATH = os.path.join(settings.BASE_DIR, 'predictor/ml_models/encoders.pkl')

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


def predict_weather(request):
    prediction = None

    if request.method == 'POST':
        data = {
            'location': request.POST['location'],
            'date': request.POST['date'],
            'mintemp': float(request.POST['mintemp']),
            'maxtemp': float(request.POST['maxtemp']),
            'raintoday': request.POST['raintoday'],
            'evaporation': float(request.POST['evaporation']),
            'sunshine': float(request.POST['sunshine']),
            'windgustdir': request.POST['windgustdir'],
            'windgustspeed': float(request.POST['windgustspeed']),
            'winddir9am': request.POST['winddir9am'],
            'winddir3pm': request.POST['winddir3pm'],
            'windspeed9am': float(request.POST['windspeed9am']),
            'windspeed3pm': float(request.POST['windspeed3pm']),
            'humidity9am': float(request.POST['humidity9am']),
            'humidity3pm': float(request.POST['humidity3pm']),
            'pressure9am': float(request.POST['pressure9am']),
            'pressure3pm': float(request.POST['pressure3pm']),
            'cloud9am': float(request.POST['cloud9am']),
            'cloud3pm': float(request.POST['cloud3pm']),
            'temp9am': float(request.POST['temp9am']),
            'temp3pm': float(request.POST['temp3pm']),
        }

        df = pd.DataFrame([data])

        date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
        df['year'] = date_obj.year
        month = date_obj.month
        df['month_sin'] = math.sin(2 * math.pi * month / 12)
        df['month_cos'] = math.cos(2 * math.pi * month / 12)

        df['pressure_delta'] = df['pressure3pm'] - df['pressure9am']
        df['temp_delta'] = df['temp3pm'] - df['temp9am']
        df['humidity_delta'] = df['humidity3pm'] - df['humidity9am']
        df['temp_range'] = df['maxtemp'] - df['mintemp']

        df['raintoday'] = 1 if data['raintoday'] == 'Yes' else 0

        cat_cols = ['location', 'windgustdir', 'winddir9am', 'winddir3pm']
        for col in cat_cols:
            try:
                df[col] = encoders[col].transform([df[col][0]])
            except:
                df[col] = 0

        feature_order = [
            'location', 'mintemp', 'maxtemp', 'evaporation', 'sunshine',
            'windgustdir', 'windgustspeed', 'winddir9am', 'winddir3pm',
            'windspeed9am', 'windspeed3pm', 'humidity9am', 'humidity3pm',
            'pressure9am', 'pressure3pm', 'cloud9am', 'cloud3pm',
            'temp9am', 'temp3pm', 'raintoday',
            'year',
            'pressure_delta', 'temp_delta', 'humidity_delta', 'temp_range',
            'month_sin', 'month_cos'
        ]

        input_data = df[feature_order]

        # 4. Dự báo
        pred_value = model.predict(input_data)[0]
        prediction = round(pred_value, 2)

    return render(request, 'index.html', {'prediction': prediction})