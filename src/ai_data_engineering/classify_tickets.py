
import json
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# 1. Load API key from .env
load_dotenv()

# 2. Initialize Gemini client
client = genai.Client()

MODEL = "gemini-3.5-flash-lite"

# 3. Sample customer support tickets
tickets = [
    {
        "ticket_id": 101,
        "message": "I was charged twice for my subscription."
    },
    {
        "ticket_id": 102,
        "message": "I cannot log into my account."
    },
    {
        "ticket_id": 103,
        "message": "How can I upgrade my subscription?"
    }
]

# 4. Define JSON output schema
CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "technical", "general"]
        }
    },
    "required": ["category"],
    "additionalProperties": False
}


# 5. Call Gemini with retry logic
def generate_with_retry(prompt, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=CATEGORY_SCHEMA,
                    max_output_tokens=100
                )
            )

            return response

        except (errors.ServerError, errors.ClientError) as e:
            # Retry temporary server/rate-limit errors only
            if e.code not in (429, 500, 502, 503, 504):
                raise

            if attempt == max_attempts - 1:
                print(f"API failed after {max_attempts} attempts.")
                raise

            # Exponential backoff with jitter
            wait = min(2 ** attempt, 8) + random.uniform(0, 1)

            print(
                f"API error {e.code}. "
                f"Retrying in {wait:.1f} seconds..."
            )

            time.sleep(wait)


# 6. Process customer tickets
results = []

for ticket in tickets:
    print(f"Processing ticket {ticket['ticket_id']}...")

    
    prompt = f"""
    You are a customer support ticket classifier.

    Classify the ticket into exactly one category:

    - billing: Payments, charges, refunds, or invoices.
    - technical: Login failures, software bugs, or errors.
    - general: Product questions, subscription upgrades,
      plan information, or general inquiries.

    Examples:
    "I was charged twice" -> billing
    "I cannot log in" -> technical
    "How can I upgrade my plan?" -> general

    Customer ticket:
    {ticket["message"]}
    """


    try:
        response = generate_with_retry(prompt)

        if not response.text:
            raise ValueError("Empty model response")

        # Parse JSON response
        data = json.loads(response.text)

        # Validate category
        category = data.get("category")

        if category not in {"billing", "technical", "general"}:
            raise ValueError(f"Invalid category: {category}")

        # Save classification result
        results.append({
            "ticket_id": ticket["ticket_id"],
            "message": ticket["message"],
            "category": category
        })

        print(f"Category: {category}")

        # Track token usage
        if response.usage_metadata:
            usage = response.usage_metadata

            print(
                f"Tokens: "
                f"input={usage.prompt_token_count or 0}, "
                f"output={usage.candidates_token_count or 0}, "
                f"total={usage.total_token_count or 0}"
            )

    except (errors.APIError, ValueError, json.JSONDecodeError) as e:
        print(f"Failed to process ticket {ticket['ticket_id']}: {e}")

        results.append({
            "ticket_id": ticket["ticket_id"],
            "message": ticket["message"],
            "category": None,
            "error": str(e)
        })

    print("-" * 40)


# 7. Print final results
print("\nFinal Classification Results:")
print(json.dumps(results, indent=2))



import pandas as pd

df = pd.DataFrame(results)

df.to_csv("classified_tickets.csv", index=False)

print("\nCSV file saved successfully!")
print(df)