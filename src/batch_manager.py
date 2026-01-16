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
except ImportError:
    pass
# DashScope import
try:
    from dashscope import BatchTextGeneration
except ImportError:
    pass

class BatchManager:
    def __init__(self, config):
        self.config = config
        self.jobs_file = config.get('batch', {}).get('metadata_file', 'outputs/batch_jobs.json')
        self.setup_clients()

    def setup_clients(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")

        if self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
        
        if self.anthropic_key:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
            
        if self.google_key:
            # Initialize official Google GenAI Client
            self.google_client = genai.Client(api_key=self.google_key)

    def load_jobs(self):
        if os.path.exists(self.jobs_file):
            with open(self.jobs_file, 'r') as f:
                return json.load(f)
        return {}

    def save_jobs(self, jobs):
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
        elif provider == "openai_compatible":
            # For Qwen (DashScope) which supports strict OpenAI Batch format
            return self._submit_openai_compatible(model_key, model_name, payloads, output_dir, model_config)
        
            
        elif provider == "qwen":
            return self._submit_qwen(model_key, model_name, payloads, output_dir)
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
                    "custom_id": p['id'],
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
        Special handler for compatible APIs (like Qwen/DashScope)
        """
        base_url = model_config.get('base_url')
        env_key = model_config.get('env_key')
        api_key = os.getenv(env_key)
        
        # Initialize a specific client for this provider
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # 1. Create JSONL
        jsonl_path = os.path.join(output_dir, f"batch_input_{model_key}.jsonl")
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for p in payloads:
                request_obj = {
                    "custom_id": p['id'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": p['system_prompt']},
                            {"role": "user", "content": p['user_query']}
                        ],
                        # Qwen sometimes requires strict parameter sets, remove temp if issues arise
                    }
                }
                writer.write(request_obj)
        
        # 2. Upload
        with open(jsonl_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
        file_id = file_response.id
        print(f"Uploaded file {file_id} to {base_url}")
        
        # 3. Create Batch
        # Note: metadata syntax might vary, keeping it simple
        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = batch_response.id
        print(f"Batch submitted: {batch_id}")
        
        # 4. Record
        jobs = self.load_jobs()
        jobs[batch_id] = {
            "provider": "openai_compatible", # To distinguish when checking
            "model_key": model_key,
            "base_url": base_url, # Needed for retrieval
            "env_key": env_key,   # Needed for retrieval
            "status": "pending",
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": output_dir,
            "input_file": jsonl_path
        }
        self.save_jobs(jobs)
        return batch_id

    def _submit_google(self, model_key, model_name, payloads, output_dir):
        # 1. Create Gemini JSONL
        jsonl_path = os.path.join(output_dir, f"batch_input_{model_key}.jsonl")
        
        # "key" is usually the ID in Gemini Batch
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for p in payloads:
                # Construct query. Since system prompts are often model-level in Gemini,
                # we can try to merge system + user or use specific fields if Batch supports it.
                # The generic format is 'contents' -> parts -> text.
                
                full_text = f"System: {p['system_prompt']}\n\nUser: {p['user_query']}"
                
                request_obj = {
                    "key": p['id'], # Custom ID
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
        # Note: Ensure model_name is valid (e.g. gemini-1.5-flash-002)
        batch_job = self.google_client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={'display_name': f"batch_run_{model_key}"}
        )
        
        batch_id = batch_job.name # format "batches/..."
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
        # Anthropic allows direct list submission (up to specific size), 
        # but large batches might need standard mechanics. 
        # The Beta Message Batches API takes a list of requests.

        requests = []
        for p in payloads:
            requests.append({
                "custom_id": p['id'],
                "params": {
                    "model": model_name,
                    "max_tokens": 1024,
                    "system": p['system_prompt'], # Anthropic system param
                    "messages": [
                        {"role": "user", "content": p['user_query']}
                    ]
                }
            })
        
        # Create Batch
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
        _submit_qwen(self, model_key, model_name, payloads, output_dir):
        # Using DashScope Python SDK (Official)
        # 1. Create JSONL File (DashScope format)
        # DashScope batch format is similar to OpenAI but we use the SDK utility or standard jsonl
        # The file needs to be uploaded or passed as path.
        
        jsonl_path = os.path.join(output_dir, f"batch_input_{model_key}.jsonl")
        
        with jsonlines.open(jsonl_path, mode='w') as writer:
            for p in payloads:
                # Construct messages
                messages = [
                    {"role": "system", "content": p['system_prompt']},
                    {"role": "user", "content": p['user_query']}
                ]
                
                # DashScope Batch Input: {"custom_id": "...", "body": {"model": "...", "input": {...}, "parameters": {...}}}
                # But typically using BatchTextGeneration.call with a file expects:
                # {"custom_id": "1", "url": "HTTP_METHOD", ...} if using OpenAI compatible
                # OR if using SDK: 
                # The SDK expects a file where each line is input to the model.
                
                # Check documentation: https://www.alibabacloud.com/help/en/model-studio/batch-inference
                # "The input file must be a JSONL file."
                # {"custom_id": "001", "body": {"model": "qwen-max", "input": {"messages": [...]}, "parameters": {}}}
                
                request_obj = {
                    "custom_id": p['id'],
                    "body": {
                        "model": model_name,
                        "input": {
                            "messages": messages
                        },
                        "parameters": {
                            "temperature": 0
                        }
                    }
                }
                writer.write(request_obj)

        print(f"Created Qwen input file {jsonl_path}")

        # 2. Upload file & Submit Job via SDK
        # DashScope SDK handles upload implicitly if path provided, or requires OSS.
        # However, the SDK `BatchTextGeneration.call` takes `input_path`.
        # If input_path is local, it might need to be uploaded to OSS first.
        # But `dashscope` starting from version 1.14 supports uploading local files automatically.
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
             raise ValueError("DASHSCOPE_API_KEY not found.")
             
        job = BatchTextGeneration.call(
            job_name=f"batch_run_{model_key}",
            input_path=jsonl_path,
            output_path=os.path.join(output_dir, "qwen_output.jsonl"), # This path is for OSS, usually ignored if accessing via ID
            api_key=api_key
        )
        
        if job.status_code == 200:
             batch_id = job.output.job_id
             print(f"Batch submitted: {batch_id}")
        else:
             raise Exception(f"DashScope submission failed: {job}")

        jobs = self.load_jobs()
        jobs[batch_id] = {
            "provider": "qwen",
            "model_key": model_key,
            "status": "pending",
            "timestamp": datetime.datetime.now().isoformat(),
            "output_dir": output_dir,
            "input_file": jsonl_path
        }
        self.save_jobs(jobs)
        return batch_id

    def self.save_jobs(jobs)
        return batch.id

    def check_and_retrieve(self):
        jobs = self.load_jobs()
        
        for batch_id, job in jobs.items():
            if job['status'] == 'completed':
                continue
            
            provider = job['provider']
            print(f"Checking {batch_id} ({provider})...")
            
            if provider == "openai":
                self._check_openai(batch_id, job, jobs)
            elif provider == "anthropic":
                self._check_anthropic(batch_id, job, jobs)
            elif provider == "google":
                self._check_google(batch_id, job, jobs)
            elif provider == "openai_compatible":
                self._check_openai_compatible(batch_id, job, jobs)
            elif provider == "qwen":
                self._check_qwen(batch_id, job, jobs)
        
        self.save_jobs(jobs)

    def _check_qwen(self, batch_id, job, jobs_dict):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        # Use simple retrieve or wait
        # The 'dashscope' library typically uses wait(job_id) which blocks, or we can check status via API
        # But SDK doesn't expose a simple 'retrieve' method easily without job object sometimes.
        # Actually checking status can usually be done via `BatchTextGeneration.call(start_time=...)` to list, 
        # OR just attempting to wait/get.
        # However, official docs suggest polling.
        
        # We can re-use BatchTextGeneration.call logic or use lower-level http
        # Let's use the .fetch() style if available or call.
        
        # It seems dashscope SDK is evolving. A common pattern:
        # job = BatchTextGeneration.get(batch_id) 
        # (This method might be fictional depending on version, let's try strict docs: 
        # "You can view the state... via the console" or using list API).
        
        # Let's assume we can fetch by job_id using the internal async client or just a GET.
        # Reverting to `requests` might be safer if SDK 'retrieve' is obscure.
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json"
        }
        # Endpoint: https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}
        url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{batch_id}"
        
        try:
            resp = requests.get(url, headers=headers)
            data = resp.json()
            # Status: PENDING, RUNNING, SUCCEEDED, FAILED
            status = data.get('output', {}).get('task_status', data.get('status'))
            
            print(f"  Status: {status}")
            
            if status == 'SUCCEEDED':
                # results url
                # often in output.results_url
                res_url = data.get('output', {}).get('results_url')
                if res_url:
                     # download
                     print(f"  Downloading from {res_url}")
                     r = requests.get(res_url)
                     output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
                     with open(output_path, 'wb') as f:
                         f.write(r.content)
                     
                     job['status'] = "completed"
                     job['result_file'] = output_path
                     print(f"  Downloaded results to {output_path}")

        except Exception as e:
            print(f"  Error checking Qwen status: {e}")

    def _check_openai(self, batch_id, job, jobs_dict):
        batch = self.openai_client.batches.retrieve(batch_id)
        current_status = batch.status
        print(f"  Status: {current_status}")
        
        if current_status == "completed" and batch.output_file_id:
            # Download results
            file_response = self.openai_client.files.content(batch.output_file_id)
            content = file_response.text
            
            output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
            with open(output_path, "w") as f:
                f.write(content)
            
            job['status'] = "completed"
            job['result_file'] = output_path
            print(f"  Downloaded results to {output_path}")

    def _check_openai_compatible(self, batch_id, job, jobs_dict):
        # Reconstruct client
        client = openai.OpenAI(api_key=os.getenv(job['env_key']), base_url=job['base_url'])
        
        batch = client.batches.retrieve(batch_id)
        current_status = batch.status
        print(f"  Status: {current_status}")
        
        if current_status == "completed" and batch.output_file_id:
            file_response = client.files.content(batch.output_file_id)
            content = file_response.text
            
            output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
            with open(output_path, "w") as f:
                f.write(content)
            
            job['status'] = "completed"
            job['result_file'] = output_path
            print(f"  Downloaded results to {output_path}")

    def _check_google(self, batch_id, job, jobs_dict):
        # Gemini check
        batch_job = self.google_client.batches.get(name=batch_id)
        # State enum mapping to string
        state = batch_job.state.name # JOB_STATE_SUCCEEDED, etc.
        print(f"  Status: {state}")
        
        if state == "JOB_STATE_SUCCEEDED":
            output_file_name = batch_job.dest.file_name
            print(f"  Downloading from {output_file_name}")
            content = self.google_client.files.download(file=output_file_name)
            
            output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
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
             # In Anthropic, checking the result URL
             if hasattr(batch, 'results_url') and batch.results_url:
                 # Depending on SDK, might need separate download steps or 'results()' method
                 # For now, let's look at iterating over results using the batch object if the SDK supports it
                 # The SDK typically exposes a method to stream results.
                 
                 results_list = []
                 for result in self.anthropic_client.messages.batches.results(batch_id):
                     results_list.append(result)
                     
                 # Save
                 output_path = os.path.join(job['output_dir'], f"batch_results_{batch_id}.jsonl")
                 # We'll save as generic jsonl
                 with jsonlines.open(output_path, mode='w') as writer:
                     for r in results_list:
                        # Serialize the Anthropic object to dict
                        writer.write(r.to_dict())

                 job['status'] = "completed"
                 job['result_file'] = output_path
                 print(f"  Downloaded results to {output_path}")

    def process_results_to_final_json(self, evaluator_instance):
        """
                        elif job['provider'] == 'qwen':
                             # Qwen Results (DashScope)
                             # Format: {"id": "...", "code": 200, "output": {"text": "...", "finish_reason": "stop"}, "request_id": "..."}
                             try:
                                 if obj.get('code') == 200:
                                     response_text = obj.get('output', {}).get('text') or obj.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content')
                                 else:
                                     response_text = f"Error: {obj.get('message')}"
                             except:
                                 response_text = "Error extracting content"
        Converts the raw downloaded batch JSONL files into the standard results.json format
        and saves them.
        """
        jobs = self.load_jobs()
        for batch_id, job in jobs.items():
            if job.get('status') == 'completed' and not job.get('processed'):
                print(f"Processing results for batch {batch_id}...")
                
                results_file = job.get('result_file')
                if not results_file or not os.path.exists(results_file):
                    print("  Result file missing.")
                    continue

                processed_results = []
                
                # Load ground truth map
                # We need the original queries to map ground truths if not present in custom_id metadata
                # Assuming custom_id == query_id
                
                query_map = {q['id']: q for q in evaluator_instance.queries}

                with jsonlines.open(results_file) as reader:
                    for obj in reader:
                        custom_id = obj.get('custom_id')
                        
                        original_query = query_map.get(custom_id)
                        if not original_query:
                            continue
                        
                        # Extract response text based on provider
                        response_text = ""
                        if job['provider'] in ['openai', 'openai_compatible']:
                            # OpenAI Batch Result Format: { "response": { "body": { "choices": [ ... ] } } }
                            try:
                                choices = obj['response']['body']['choices']
                                response_text = choices[0]['message']['content']
                            except:
                                response_text = "Error extracting content"
                        elif job['provider'] == 'anthropic':
                            # Anthropic Result: { "result": { "message": { "content": [...] } } }
                            try:
                                content_list = obj['result']['message']['content']
                                response_text = content_list[0]['text']
                            except:
                                response_text = "Error extracting content"
                        elif job['provider'] == 'google':
                            # Gemini Batch Result
                            try:
                                # candidates -> content -> parts -> text
                                cand = obj['response']['candidates'][0]
                                parts = cand['content']['parts']
                                response_text = parts[0]['text']
                            except:
                                response_text = "Error extracting content"

                        generated_sparql = evaluator_instance.clean_sparql(response_text)
                        
                        result_record = {
                            "id": custom_id,
                            "query": original_query['query'],
                            "model": job['model_key'],
                            "generated_sparql": generated_sparql,
                            "ground_truth_sparql": original_query.get('ground_truth_sparql'),
                            "generated_count": -1,
                            "ground_truth_count": -1,
                            "is_match": False,
                            "execution_error": "Execution Skipped",
                            "raw_llm_response": response_text
                        }
                        processed_results.append(result_record)
                
                # Save final JSON
                final_path = os.path.join(job['output_dir'], "results_summary_batch.json")
                with open(final_path, 'w') as f:
                    json.dump(processed_results, f, indent=2)
                
                print(f"  Final results saved to {final_path}")
                job['processed'] = True
        
        self.save_jobs(jobs)
