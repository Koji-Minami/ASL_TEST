import os,base64,json
from dotenv import load_dotenv
from anthropic import AnthropicVertex

load_dotenv()
REGION = os.getenv('REGION')  # Ensure this is set correctly in .env
PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT') # Ensure this is set correctly in .env

LOCATION = "us-central1" #Double-check this against Anthropic's documentation.

client = AnthropicVertex(region=LOCATION, project_id=PROJECT) # Using PROJECT variable

message = client.messages.create(
  max_tokens=1024,
  messages=[
    {
      "role": "user",
      "content": "Send me a recipe for banana bread.",
    }
  ],
  model="claude-3-5-sonnet-v2@20241022",
)
print(message.model_dump_json(indent=2))