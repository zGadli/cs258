import requests
import json

url = "http://0.0.0.0:8080/prime"

payload = {"number" : 10}

response = requests.post(url, json.dumps(payload))

print(response.text)

# Or you can use Postman.
# https://www.postman.com/