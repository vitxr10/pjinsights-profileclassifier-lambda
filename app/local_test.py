import json
from main import lambda_handler

with open("../events/event.json", "r", encoding="utf-8") as f:
    event = json.load(f)

response = lambda_handler(event, None)
print(response)