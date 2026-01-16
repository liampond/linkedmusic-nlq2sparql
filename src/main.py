import argparse
from src.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="LinkedMusic NLQ to SPARQL Evaluator")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Run a specific model only")
    parser.add_argument("--batch", action="store_true", help="Use Batch API where available")
    parser.add_argument("--check-batch", action="store_true", help="Check status of pending batches and retrieve results")
    parser.add_argument("--filter-db", help="Filter queries by target database (e.g. DIAMM)")
    args = parser.parse_args()

    evaluator = Evaluator(args.config)
    
    # Filter queries if requested
    if args.filter_db:
        original_count = len(evaluator.queries)
        evaluator.queries = [q for q in evaluator.queries if args.filter_db in q.get('target_databases', '')]
        print(f"Filtered queries from {original_count} to {len(evaluator.queries)} (Filter: {args.filter_db})")

    if args.check_batch:
        from src.batch_manager import BatchManager
        bm = BatchManager(evaluator.config)
        bm.check_and_retrieve()
        bm.process_results_to_final_json(evaluator)
        return

    if args.batch:
        # Batch Submission Mode
        from src.batch_manager import BatchManager
        bm = BatchManager(evaluator.config)
        
        if args.model:
            models_to_run = [args.model]
        else:
            models_to_run = evaluator.config['models'].keys()

        payloads = evaluator.generate_prompt_payloads()
        
        for model in models_to_run:
            try:
                 # Create output dir for this run
                output_dir = evaluator.prepare_output_dir(model)
                evaluator.save_run_metadata(output_dir, model)
                
                bm.submit_batch(model, payloads, output_dir)
            except Exception as e:
                print(f"FAILED to submit batch for {model}: {e}")
                # Continue to next model
        
        return

    # Interactive Mode (Normal)
    if args.model:
        evaluator.run_evaluation_for_model(args.model)
    else:
        evaluator.run_all()

if __name__ == "__main__":
    main()
