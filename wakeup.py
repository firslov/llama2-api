import requests
import time
import json


url = "http://0.0.0.0:666/v1/chat/completions"

while True:
    time.sleep(1500)
    msg = json.dumps(
        {"messages": [{"role": "user", "content": "hello"}], "stream": True}
    )
    res = requests.post(url, msg)
