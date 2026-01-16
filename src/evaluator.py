import os
import json
import re
# import pandas as pd # Unused
import yaml
from tqdm import tqdm
from src.utils import load_config, load_ontology, setup_env
from src.llm_manager import LLMManager
from src.sparql_client import SPARQLClient
import datetime

class Evaluator:
    def __init__(self, config_path="config.yaml"):
        setup_env()
        self.config = load_config(config_path)
        self.llm_manager = LLMManager(self.config)
        # self.sparql_client = SPARQLClient(self.config['sparql_endpoint']) # Disabled as requested
        self.sparql_client = None 
        
        self.ontology_content = load_ontology(self.config['ontology_file'])
        
        with open(self.config['input_data'], 'r') as f:
            self.queries = json.load(f)

    def generate_prompt_payloads(self):
        """
        Generates the system prompt and user query for all items.
        Returns a list of dicts: {'id': ..., 'system_prompt': ..., 'user_query': ...}
        """
        payloads = []
        for item in self.queries:
            nl_query = item['query']
            query_id = item['id']
            system_prompt = self.construct_system_prompt(nl_query)
            payloads.append({
                "id": query_id,
                "system_prompt": system_prompt,
                "user_query": nl_query,
                "ground_truth": item.get('ground_truth_sparql')
            })
        return payloads

    def clean_sparql(self, raw_response):
        """
        Extracts SPARQL code from a raw LLM response (handling markdown blocks).
        """
        # Look for code blocks
        code_block_pattern = r"```sparql(.*?)```"
        matches = re.findall(code_block_pattern, raw_response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        
        # Fallback: look for generic code blocks
        code_block_pattern_generic = r"```(.*?)```"
        matches_generic = re.findall(code_block_pattern_generic, raw_response, re.DOTALL)
        if matches_generic:
            return matches_generic[0].strip()

        # Fallback: if no blocks, assume the whole text is SPARQL (risky but possible)
        # Or look for common SPARQL keywords to trim prelude
        
        return raw_response.strip()

    def construct_system_prompt(self, nl_query):
        # Load the system prompt template
        if os.path.exists("system_prompt.txt"):
             with open("system_prompt.txt", "r") as f:
                template = f.read()
        else:
            # Fallback (though we expect the file to exist now)
            template = """I have a graph database containing musical linked data...
Please write me a SPARQL query to perform the following query:
Find all compositions in DIAMM that are composed by Guillaume de Machaut
..."""
        
        # Replace the placeholder query with the actual user query
        # We look for the specific placeholder line provided in the instructions
        placeholder = "Find all compositions in DIAMM that are composed by Guillaume de Machaut"
        
        if placeholder in template:
            prompt = template.replace(placeholder, nl_query)
        else:
            # If explicit placeholder not found, just append/prepend or verify template format
            # But let's assume it works as instructed.
            # If not found, we insert it into the prompt explicitly.
            prompt = template + f"\n\nPlease write a SPARQL query for: {nl_query}"

        # Append ontology at the end
        prompt += f"\n\nOntology Definitions:\n{self.ontology_content}"
        
        return prompt

    def prepare_output_dir(self, model_key):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"run_{timestamp}_{model_key}"
        output_path = os.path.join(self.config['output_dir'], dir_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    def save_run_metadata(self, output_dir, model_key):
        # Save config snapshot
        with open(os.path.join(output_dir, "config_snapshot.yaml"), "w") as f:
            yaml.dump(self.config, f)
        
        # Save Ontology snapshot
        with open(os.path.join(output_dir, "ontology_snapshot.ttl"), "w") as f:
            f.write(self.ontology_content)

    def run_evaluation_for_model(self, model_key):
        print(f"Starting evaluation for model: {model_key}")
        
        # Setup specific output directory for this run
        run_output_dir = self.prepare_output_dir(model_key)
        self.save_run_metadata(run_output_dir, model_key)
        print(f"Run output directory: {run_output_dir}")

        results = []
        logs_path = os.path.join(run_output_dir, "detailed_logs.jsonl")
        
        for item in tqdm(self.queries):
            nl_query = item['query']
            ground_truth = item.get('ground_truth_sparql')
            query_id = item['id']
            
            system_prompt = self.construct_system_prompt(nl_query)

            # 1. Get LLM Response
            llm_response = self.llm_manager.get_response(model_key, nl_query, system_prompt)
            generated_sparql = self.clean_sparql(llm_response)

            # SKIP Execution for now
            gen_count = -1
            execution_error = "Execution Skipped (Endpoint Unavailable)"
            gt_count = -1
            is_match = False
            
            # Record detailed log for debugging/linking
            log_entry = {
                "id": query_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "model": model_key,
                "user_query": nl_query,
                "system_prompt": system_prompt, # Full prompt context
                "full_response": llm_response,
                "extracted_sparql": generated_sparql
            }
            
            # Append to JSONL log
            with open(logs_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            result_record = {
                "id": query_id,
                "query": nl_query,
                "model": model_key,
                "generated_sparql": generated_sparql,
                "ground_truth_sparql": ground_truth,
                "generated_count": gen_count, # -1
                "ground_truth_count": gt_count, # -1
                "is_match": is_match, # False
                "execution_error": execution_error,
                "raw_llm_response": llm_response
            }
            results.append(result_record)

        # Save results
        filename = f"{run_output_dir}/results_summary.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved results to {filename}")
        
        # Calculate accuracy (will be 0 for now)
        matches = sum(1 for r in results if r['is_match'])
        total = len(results)
        print(f"Accuracy for {model_key}: {matches}/{total} ({matches/total*100:.2f}%)")

    def run_all(self):
        models = self.config['models'].keys()
        for model in models:
            try:
                self.run_evaluation_for_model(model)
            except Exception as e:
                print(f"Failed to run evaluation for {model}: {e}")
