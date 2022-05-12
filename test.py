import requests
import sys
import base64

imgb64 = open(sys.argv[1], 'rb').read()
imgb64 = base64.b64encode(imgb64).decode()
url = 'http://localhost:8080/predict'
obj = {'image':imgb64}
x = requests.post(url, json=obj)
print(x.text)