import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'min_soporte':2, 'min_confianza':9, 'mini_lift':6})
r2 = requests.post(url,json={'objeto1','objeto2'})
r3 = requests.post(url,json={'metodo'})
print(r.json())