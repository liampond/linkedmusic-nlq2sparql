# LinkedMusic NLQ to SPARQL Evaluator

This project evaluates the performance of Large Language Models (LLMs) in converting Natural Language Queries (NLQ) to SPARQL queries for the LinkedMusic RDF dataset.

## Project Structure

- `src/`: Source code for the evaluator, LLM manager, and SPARQL client.
- `data/`: Contains the input queries and the ontology file.
- `outputs/`: Stores the JSON results of the evaluation runs.
- `config.yaml`: Configuration for model names, API keys, and endpoints.

## Setup

1. **Install Dependencies**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory (or set them in your shell) with your API keys:
   ```bash
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...
   DASHSCOPE_API_KEY=... # For Qwen if using Dashscope/OpenAI Compatible
   ```

3. **Data Preparation**
   - **Queries:** Populate `data/queries.json` with your test cases. Format:
     ```json
     [
       {
         "id": "1",
         "query": "Find all symphonies by Beethoven",
         "ground_truth_sparql": "SELECT ?s WHERE { ... }"
       }
     ]
     ```
   - **Ontology:** Update `data/ontology.ttl` with the relevant RDF schema definitions.

4. **Configuration**
   Edit `config.yaml` to change model versions or the SPARQL endpoint URL.

## Usage

Run the evaluation for all configured models:
```bash
python -m src.main
```

Run for a specific model (key must match `config.yaml` keys):
```bash
python -m src.main --model openai
python -m src.main --model claude
```

## Detailed Results
Results are saved to the `outputs/` directory as JSON files, containing the generated SPARQL, execution counts, and match status for every query.
