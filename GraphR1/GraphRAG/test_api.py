import requests
import time
import json
import pdb


url = "http://0.0.0.0:8000/search"

print("⏳ Waiting for API to start...")

# 测试多个 query
data = {
    "queries": [
        "Which actor does American Beauty and American Beauty have in common?"
    ]
}

print("📡 Sending request...")

response = requests.post(url, json=data)

print("Status:", response.status_code)

try:
    parsed = response.json()
except json.JSONDecodeError:
    print("❌ Response not valid JSON:")
    print(response.text)
    raise

print("Response JSON:")
print(json.dumps(parsed, indent=2))
