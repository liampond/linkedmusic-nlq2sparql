import os
import json
import jsonlines
import time
import datetime
from typing import Dict, List
import openai
import anthropic

# Gemini import
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

class BatchManager:
    def __init__(self, config):
        self.config = config
        # Default to raw_outputs instead of outputs
        default_dir = config.get('output_dir', 'raw_outputs')
        self.jobs_file = config.get('batch', {}).get('metadata_file', os.path.join(default_dir, 'batch_jobs.json'))
        self.setup_clients()

    def setup_clients(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")

        if self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
        
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            
        if self.google_key and HAS_GOOGLE:
            # Initialize official Google GenAI Client
            self.google_client = genai.Client(api_key=self.google_key)
        elif self.google_key and not HAS_GOOGLE:
            print("Warning: GOOGLE_API_KEY found but google-genai not installed.")

    def load_jobs(self):
        if os.path.exists(self.jobs_file):
            with open(self.jobs_file, 'r') as f:
                return json.load(f)
        return {}

    def save_jobs(self, jobs):
        dir_path = os.path.dirname(self.jobs_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2)

    def submit_batch(self, model_key: str, payloads: List[Dict], output_dir: str):
        """
        Dispatches batch submission based on provider.
        """
        model_config = self.config['models'].get(model_key)
        provider = model_config.get('provider')
        model_name = model_config.get('model_name')
        
        print(f"Submitting batch for {model_key} ({provider})...")

        if provider == "openai": 
            return self._submit_openai(model_key, model_name, payloads, output_dir)
        elif provider == "anthropic":
            return self._submit_anthropic(model_key, model_name, payloads, output_dir)
        
        elif provider == "google":
            return self._submit_google(model_key, model_name, payloads, output_dir)
        
        else:
            print(f"Batch submission not yet implemented for provider: {provider}")
            return None

    def _submit_openai(self, model_key, model_name, payloads, output_dir):
        # 1. Create JSONL file
        jsonl_path = os.path.join(output_dir, f"batch_input_{model_key}.jsonl")
        
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for p in payloads:
                # OpenAI Batch Format
                request_obj = {
                    "custom_id": str(p['id']),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": p['system_prompt']},
                            {"role": "user", "content": p['user_query']}
                        ],
                        "temperature": 0
                    }
                }
                writer.write(request_obj)
        
        # 2. Upload File
        with open(jsonl_path, "rb") as f:
            file_response = self.openai_client.files.create(file=f, purpose="batch")
        
        file_id = file_response.id
        print(f"Uploaded file {file_id}")

        # 3. Create Batch
        batch_response = self.openai_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        batch_id = batch_response.id
        print(f"Batch submitted: {batch_id}")

        # 4. Record Job
        jobs = self.load_jobs()
        jobs[batch_id] = {
            "provider": "openai",
            "model_key": model_key,
            "status": "pending",
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": output_dir,
            "input_file": jsonl_path
        }
        self.save_jobs(jobs)
        return batch_id

    def _submit_openai_compatible(self, model_key, model_name, payloads, output_dir, model_config):
        """
        Special handler for compatible APIs (like Qwen/DashScope using OpenAI SDK)
        """
        pass

    def _submit_google(self, model_key, model_name, payloads, output_dir):
        # 1. Create Gemini JSONL
        jsonl_path = os.path.join(output_dir, f"batch_input_{model_key}.jsonl")
        
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for p in payloads:
                full_text = f"System: {p['system_prompt']}\n\nUser: {p['user_query']}"
                
                request_obj = {
                    "custom_id": str(p['id']),
                    "method": "generateContent",
                    "request": {
                        "contents": [
                            {"parts": [{"text": full_text}]}
                        ]
                    }
                }
                writer.write(request_obj)
        
        print(f"Created Gemini input file {jsonl_path}")

        # 2. Upload File
        uploaded_file = self.google_client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(display_name=f"batch_input_{model_key}", mime_type="application/json")
        )
        print(f"Uploaded file uri: {uploaded_file.uri}")
        
        # 3. Submit Batch
        batch_job = self.google_client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={'display_name': f"batch_run_{model_key}"}
        )
        
        batch_id = batch_job.name
        print(f"Batch submitted: {batch_id}")
        
        jobs = self.load_jobs()
        jobs[batch_id] = {
            "provider": "google",
            "model_key": model_key,
            "status": "pending",
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": output_dir,
            "input_file": jsonl_path
        }
        self.save_jobs(jobs)
        return batch_id

    def _submit_anthropic(self, model_key, model_name, payloads, output_dir):
        requests = []
        for p in payloads:
            requests.append({
                "custom_id": str(p['id']),
                "params": {
                    "model": model_name,
                    "max_tokens": 1024,
                    "system": p['system_prompt'],
                    "messages": [
                        {"role": "user", "content": p['user_query']}
                    ]
                }
            })
        
        batch = self.anthropic_client.messages.batches.create(
            requests=requests
        )
        
        print(f"Batch submitted: {batch.id}")
        
        jobs = self.load_jobs()
        jobs[batch.id] = {
            "provider": "anthropic",
            "model_key": model_key,
            "status": "pending",
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": output_dir
        }
        self.save_jobs(jobs)
        if not HAS_GOOGLE:
             print("Error: google-genai package not installed, cannot submit Gemini batch.")
             return None
             
        return batch.id

    def check_and_retrieve(self):
        jobs = self.load_jobs()
        
        for batch_id, job in jobs.items():
            if job.get('status') == 'completed':
                continue
            
            provider = job['provider']
            print(f"Checking {batch_id} ({provider})...")
            
            try:
                if provider == "openai":
                    self._check_openai(batch_id, job, jobs)
                elif provider == "anthropic":
                    self._check_anthropic(batch_id, job, jobs)
                elif provider == "google":
                    self._check_google(batch_id, job, jobs)
            except Exception as e:
                print(f"Error checking status for {batch_id}: {e}")
        
        self.save_jobs(jobs)

    def _check_openai(self, batch_id, job, jobs_dict):
        batch = self.openai_client.batches.retrieve(batch_id)
        current_status = batch.status
        print(f"  Status: {current_status}")
        
        if current_status == "completed" and batch.output_file_id:
            file_response = self.openai_client.files.content(batch.output_file_id)
            content = file_response.text
            
            output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
            with open(output_path, "w") as f:
                f.write(content)
            
            job['status'] = "completed"
            job['result_file'] = output_path
            print(f"  Downloaded results to {output_path}")

    def _check_google(self, batch_id, job, jobs_dict):
        batch_job = self.google_client.batches.get(name=batch_id)
        state = batch_job.state.name
        print(f"  Status: {state}")
        
        if state == "JOB_STATE_SUCCEEDED":
            output_file_name = batch_job.dest.file_name
            print(f"  Downloading from {output_file_name}")
            content = self.google_client.files.download(file=output_file_name)
            
            # Sanitize batch_id for filesystem (Gemini IDs contain '/')
            safe_batch_id = batch_id.replace("/", "_")
            output_path = os.path.join(job['output_dir'], f"batch_results_{safe_batch_id}.jsonl")
            with open(output_path, "wb") as f:
                f.write(content)
                
            job['status'] = "completed"
            job['result_file'] = output_path
            print(f"  Downloaded results to {output_path}")
    
    def _check_anthropic(self, batch_id, job, jobs_dict):
        batch = self.anthropic_client.messages.batches.retrieve(batch_id)
        current_status = batch.processing_status
        print(f"  Status: {current_status}")
        
        if current_status == "ended":
             # Iterating over results
             results = []
             try:
                 for result in self.anthropic_client.messages.batches.results(batch_id):
                     results.append(result.to_dict()) # Ensure dict serializable
             except Exception as e:
                 print(f"Error fetching Anthropic results: {e}")
                 # Try simple list fallback if SDK allows
                 pass

             output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
             with open(output_path, "w") as f:
                 for r in results:
                     f.write(json.dumps(r) + "\n")
            
             job['status'] = "completed"
             job['result_file'] = output_path
             print(f"  Downloaded results to {output_path}")

    def process_results_to_final_json(self, evaluator):
        """
        Processes completed batch jobs and converts their raw output 
        to the standard results_summary.json format used by the Evaluator.
        """
        jobs = self.load_jobs()
        processed_count = 0
        
        for batch_id, job in jobs.items():
            if job.get('status') == 'completed' and not job.get('results_processed'):
                print(f"Processing results for batch {batch_id}...")
                
                result_file = job.get('result_file')
                if not result_file or not os.path.exists(result_file):
                    print(f"  Result file missing: {result_file}")
                    continue
                    
                model_key = job['model_key']
                output_dir = job['output_dir']
                
                query_map = {str(q['id']): q for q in evaluator.queries}
                
                results = []
                logs_path = os.path.join(output_dir, "detailed_logs.jsonl")
                
                with jsonlines.open(result_file) as reader:
                    for line in reader:
                        provider = job['provider']
                        custom_id = None
                        response_text = ""
                        
                        try:
                            if provider in ["openai"]:
                                custom_id = line.get('custom_id')
                                choice = line.get('response', {}).get('body', {}).get('choices', [{}])[0]
                                response_text = choice.get('message', {}).get('content', "")
                            elif provider == "anthropic":
                                custom_id = line.get('custom_id')
                                result = line.get('result', {})
                                if result.get('type') == 'succeeded':
                                    content_list = result.get('message', {}).get('content', [])
                                    if content_list:
                                        response_text = content_list[0].get('text', "")
                            elif provider == "google":
                                custom_id = line.get('custom_id') 
                                # Gemini Batch Output often matches input structure or uses custom_id if provided 
                                # In the newest API, check 'custom_id' field.
                                if not custom_id and 'request' in line:
                                     # Sometimes it echoes request
                                     pass
                                # Response part
                                try:
                                    response_text = line['response']['candidates'][0]['content']['parts'][0]['text']
                                except:
                                    pass

                        except Exception as e:
                            print(f"  Error parsing line: {e}")
                            continue

                        if custom_id and custom_id in query_map:
                            item = query_map[custom_id]
                            nl_query = item['query']
                            generated_sparql = evaluator.clean_sparql(response_text)
                            
                            # Log
                            log_entry = {
                                "id": custom_id,
                                "timestamp": datetime.datetime.now().isoformat(),
                                "model": model_key,
                                "user_query": nl_query,
                                "full_response": response_text,
                                "extracted_sparql": generated_sparql
                            }
                            with open(logs_path, "a") as f:
                                f.write(json.dumps(log_entry) + "\n")
                                
                            result_record = {
                                "id": custom_id,
                                "query": nl_query,
                                "model": model_key,
                                "generated_sparql": generated_sparql,
                                "ground_truth_sparql": item.get('ground_truth_sparql'),
                                "generated_count": -1,
                                "ground_truth_count": -1,
                                "is_match": False,
                                "execution_error": "Execution Skipped (Batch Mode)",
                                "raw_llm_response": response_text
                            }
                            results.append(result_record)
                
                summary_path = os.path.join(output_dir, "results_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                job['results_processed'] = True
                processed_count += 1
                
        if processed_count > 0:
            self.save_jobs(jobs)
            print(f"Processed {processed_count} completed batches.")
        else:
            print("No new completed batches to process.")
