import requests
diabvals = {'model': 1, 'np': 6, "pgc": 148, "dpb": 72, "tst": 35, "si": 0, "bmi": 33.6, "dpf": 0.627, "age": 50}
canvals = {'model': 2, 'age': 48, "bmi": 23.5, "gluc": 70, "inc": 2.707, "HOMA": 0.467408667, "lep": 8.8071, "adi": 9.7024, "res": 7.99585, "mcp": 417.114}
heartvals = {'model': 2, 'age': 67, "sex": 1.0, "gluc": 4.0, "inc": 160.0, "HOMA": 286.0, "lep": 0.0, "adi": 108.0, "res": 1.0, "mcp": 1.5, "mcp": 2.0, "mcp": 3.0, "mcp": 3.0}



result = requests.get("http://localhost:5000/", params = diabvals)
print(result.json())

result = requests.get("http://localhost:5000/", params = canvals)
print(result.json())

result = requests.get("http://localhost:5000/", params = heartvals)
print(result.json())