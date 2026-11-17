# AI Data Engineering

A hands-on Python learning project for exploring how generative AI fits into data engineering workflows. The current examples use Google Gemini to classify customer support tickets, validate structured responses, and export the results to CSV.

## What's included

- **Ticket classification:** Assigns sample support tickets to `billing`, `technical`, or `general` using a JSON response schema.
- **Response validation:** Parses the model's JSON and checks that each category is allowed.
- **Retry handling:** Retries rate limits and selected server errors up to three times with exponential backoff and jitter.
- **Usage reporting:** Prints input, output, and total token counts when available.
- **CSV export:** Saves successful classifications and per-ticket errors using pandas.
- **Gemini smoke test:** Sends a short prompt asking Gemini to explain ETL versus ELT.

This repository is a learning workspace with runnable examples. The classifier currently uses three hardcoded tickets.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- A Google Gemini API key with access to the model configured in the scripts

## Setup

From the repository root, install the dependencies:

```sh
uv sync
```

Create a `.env` file in the repository root:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

The scripts load this file with `python-dotenv`. `.env` is excluded from Git.

Both Gemini scripts currently set the model to `gemini-3.5-flash-lite`. If your account uses a different model, update the model name in each script before running it.

### If you use pyenv

The `.python-version` file selects Python `3.12`. If pyenv reports that this version is missing, install it:

```sh
pyenv install 3.12
```

## Run the examples

Run these commands from the repository root.

Check the Gemini connection:

```sh
uv run python -m ai_data_engineering.gemini_test
```

Classify the sample support tickets:

```sh
uv run python -m ai_data_engineering.classify_tickets
```

The classifier prints progress, categories, token usage, and final results. It writes `classified_tickets.csv` to the current working directory, replacing an existing file with that name.

Example output:

```csv
ticket_id,message,category
101,I was charged twice for my subscription.,billing
102,I cannot log into my account.,technical
103,How can I upgrade my subscription?,general
```

If a ticket fails during processing, its category is empty and an `error` column records the failure. Model responses may vary.

The package also includes a starter command that prints a greeting:

```sh
uv run ai-data-engineering
```

## Project structure

```text
.
├── pyproject.toml                  # Package metadata and dependencies
├── .python-version                # Project Python version
└── src/
    └── ai_data_engineering/
        ├── __init__.py             # Starter CLI entry point
        ├── gemini_test.py          # Basic Gemini request
        ├── classify_tickets.py     # Classification and CSV export
        └── classified_tickets.csv  # Saved sample results
```

## Dependencies

| Package | Purpose |
| --- | --- |
| `google-genai` | Gemini API client used by the examples |
| `pandas` | Tabular results and CSV export |
| `python-dotenv` | Load local environment variables |
| `openai` | Installed for OpenAI experiments; not currently used by the examples |

## Try extending it

- Replace the hardcoded tickets with records from a CSV file or database.
- Add categories and update the prompt, response schema, and validation together.
- Compare predicted categories against labeled examples to measure accuracy.
- Explore batching, logging, and alternative model providers.
