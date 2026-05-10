import requests

url = "https://shl-assessment-y2q2.onrender.com/chat"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Hiring a mid-level Java developer with 4-5 years experience and stakeholder management skills"
        }
    ]
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())