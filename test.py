import requests
import sys
import base64

imgb64 = open(sys.argv[1], 'rb').read()
#print(imgb64)
imgb64 = base64.b64encode(imgb64).decode()
#print(imgb64)
url = 'http://localhost:3001/predict'
obj = {'image':imgb64}
x = requests.post(url, json=obj)
print(x.text)