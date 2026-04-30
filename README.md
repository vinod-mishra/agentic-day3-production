# Day 3 Assignment – Agentic AI Bootcamp
### Prompting That Ships – Production Hardening

A minimal but production-hardened customer support agent built with **LangChain + Google Generative AI**.

---

## Features

| # | Feature | Where in code |
|---|---------|---------------|
| 1 | **Prompts as Code** – system prompt owned by YAML, not Python | `prompts/support_agent_v1.yaml` + `load_yaml_prompt()` |
| 2 | **3-Layer Injection Defense** – input scan → hardened prompt → output scan | `detect_injection()`, `safe_agent_invoke()` |
| 3 | **Production Error Handling** – retries + exponential backoff + error categories | `production_invoke()` |
| 4 | **Circuit Breaker** – blocks LLM calls after repeated failures | `CircuitBreaker`, `guarded_invoke()` |
| 5 | **Session Cost Tracker** – budget enforcement + structured logging | `SessionCostTracker`, `budget_aware_invoke()` |

---

## Project Structure

```
agentic-day3-production/
├── app.py                        ← main entry point (run this)
├── prompts/
│   └── support_agent_v1.yaml    ← system prompt (Prompts as Code)
├── requirements.txt             ← Python dependencies
├── .env                         ← your API key (not committed)
├── .gitignore                   ← .env is excluded here
└── README.md                    ← this file
```

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd agentic-day3-production

# 2. Create a virtual environment (recommended)
conda create --prefix ./env python=3.12 -y
conda activate ./env 

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
# Create a .env file in the project root and add your Google API key:
# GOOGLE_API_KEY=your_actual_api_key_here
```

> ⚠️ **WARNING: Never commit `.env` to Git.** It contains your secret API key.
> The `.gitignore` already excludes `.env`, but always double-check before pushing:
> ```bash
> git status   # .env must NOT appear here
> ```

---

## Running the App

```bash
python app.py
```

Expected output:
```
Normal query response:  <LLM answer about the refund policy>
Injection attempt blocked by detect_injection.
Total calls: 1
Total cost (USD): 0.0000225
Budget remaining (USD): 0.4999775
```

---

## How Each Production Feature Works

**Prompts as Code** – The agent's personality and constraints live in `prompts/support_agent_v1.yaml`. Changing agent behaviour means editing the YAML, not the Python code.

**Injection Defense** – Three layers: (1) regex scan of the raw user input, (2) the YAML system prompt itself instructs the model to refuse override attempts, (3) the model's output is scanned for dangerous markers before being returned.

**Error Handling** – `production_invoke()` wraps every LLM call. Rate-limit errors trigger exponential back-off (2 s → 4 s → 8 s). Context-overflow errors fail fast. All outcomes are returned as a typed `InvocationResult`.

**Circuit Breaker** – After 5 consecutive failures the circuit opens and all further requests are blocked immediately, preventing cascading failures. After a 60-second cooldown it moves to half-open and allows one trial request.

**Cost Tracking** – `SessionCostTracker` logs every call as structured JSON and enforces a per-session USD budget. When the budget is exceeded, new requests are rejected before hitting the LLM.