from google import genai

client = genai.Client(api_key="AIzaSyDubjv8Aq9H8XgCop3rEPj3gniA2lsMHTg")

models = client.models.list()

for m in models:
    print(m.name)