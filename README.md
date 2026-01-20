# LinkedMusic NLQ to SPARQL Evaluator

This project evaluates the performance of Large Language Models (LLMs) in converting Natural Language Queries (NLQ) to SPARQL queries for the LinkedMusic RDF dataset.

## Project Structure

- `src/`: Source code for the evaluator, LLM manager, and SPARQL client.
- `ontologies/`: Contains the split ontology files.
- `queries.json`: Contains the input queries.
- `raw_outputs/`: Stores the raw JSON results from Batch APIs.
- `outputs/`: Stores the clean, generated SPARQL files organized by model and trial.
- `config.yaml`: Configuration for model names, API keys, and endpoints.
- `trials_map.json`: Maps trial numbers to their output files.

## Setup

1. **Install Dependencies**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   ```

3. **Data Preparation**
   - **Queries:** `queries.json` contains the evaluation set.
   - **Ontology:** `ontologies/combined.ttl` is generated from individual files in `ontologies/`.

## Usage

### 1. Batch Submission (Recommended)
Submit batch jobs for all configured models:
```bash
python -m src.main --batch
```

### 2. Check Status & Retrieve Results
Check pending batches and download completed results:
```bash
python -m src.main --check-batch
```

### 3. Organize SPARQL Outputs
Once batches are downloaded, organize the SPARQL into the clean folder structure:
```bash
python organize_sparql.py
```


Run for a specific model (key must match `config.yaml` keys):
```bash
python -m src.main --model openai
python -m src.main --model claude
```

## Detailed Results
Results are saved to the `outputs/` directory as JSON files, containing the generated SPARQL, execution counts, and match status for every query.
