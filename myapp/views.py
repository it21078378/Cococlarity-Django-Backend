from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pickle
import pandas as pd
import shap  # Make sure SHAP is installed
from pytrends.request import TrendReq
import requests




import time
from random import randint

# Load the trained model
with open('ExportedPredictionModel.pickle', 'rb') as file:
    model = pickle.load(file)

# Initialize the Google Trends API
pytrends = TrendReq(hl='en-US', tz=360)

# Define the feature names for your model
feature_names = ['inflation', 'CAGR_x', 'lkr_to_usd_exchange_rate', 'fire_sum', 'floods_count', 'coconut_oil_price_per_metric_ton', 'num_exporters']

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = request.POST.dict()

        # Convert POST data into a DataFrame
        inflation_data = pd.DataFrame({'inflation': [float(data['inflation'])], 
                                       'CAGR_x': [float(data['CAGR_x'])], 
                                       'lkr_to_usd_exchange_rate': [float(data['lkr_to_usd_exchange_rate'])], 
                                       'fire_sum': [float(data['fire_sum'])],
                                       'floods_count': [float(data['floods_count'])],
                                       'coconut_oil_price_per_metric_ton': [float(data['coconut_oil_price_per_metric_ton'])],
                                       'num_exporters': [int(data['num_exporters'])]})

        # Make predictions
        predicted_categories = model.predict(inflation_data)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)  # Use the appropriate explainer for your model type
        shap_values = explainer.shap_values(inflation_data)

        # Convert SHAP values to a dictionary with lists
        shap_values_dict = {feature_names[i]: shap_values[0][i].tolist() for i in range(len(feature_names))}

        # Construct the response
        predictions = {
            'prediction': predicted_categories[0],  # Assuming single prediction
            'shap_values': shap_values_dict
        }

        return JsonResponse(predictions, safe=False)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed.'})

@csrf_exempt
def google_trends(request):
    if request.method == 'POST':
        data = request.POST.dict()

        if 'search_term' in data:
            search_term = data['search_term']
            timeframe = data.get('timeframe', 'today 12-m')  # Default timeframe

            attempt = 0
            max_attempts = 5

            while attempt < max_attempts:
                try:
                    # Build the payload and make the request
                    pytrends.build_payload([search_term], cat=0, timeframe=timeframe, geo='', gprop='')

                    # Get interest by region
                    interest_by_region = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)
                    interest_by_region = interest_by_region.sort_values(by=search_term, ascending=False)
                    interest_by_region.reset_index(inplace=True)
                    interest_data = interest_by_region.to_dict(orient='records')

                    return JsonResponse({'interest_by_region': interest_data}, safe=False)

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # HTTP 429 Too Many Requests
                        attempt += 1
                        time.sleep(2 ** attempt + randint(0, 1000) / 1000)  # Exponential backoff with random jitter
                    else:
                        return JsonResponse({'error': f'Request failed: {str(e)}'}, status=e.response.status_code)

            return JsonResponse({'error': 'Too many requests. Please try again later.'}, status=429)

        return JsonResponse({'error': 'No search term provided.'}, status=400)

    return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)