from google import genai

client = genai.Client(api_key="AIzaSyDcG1RR5PjwOdVLUJhNss2aw5GXg7qWUmw")

models = client.models.list()

for m in models:
    print(m.name)