import os
import json
import shutil

# Configuration
# Read from trials_map.json
with open('trials_map.json', 'r') as f:
    TRIALS = json.load(f)

# Cast text keys "1", "2" back to integers if needed or just use str keys?
# My previous code used integer iteration: `for trial_num, providers in TRIALS.items():`
# JSON keys are always strings.
# I should convert them.

TRIALS = {int(k): v for k, v in TRIALS.items()}

OUTPUT_DIR = "outputs"

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for trial_num, providers in TRIALS.items():
        for provider, json_path in providers.items():
            if not os.path.exists(json_path):
                print(f"Skipping {provider} Trial {trial_num}: file not found {json_path}")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON at {json_path}")
                continue
            
            # Create directory: generated_sparql/{provider}/trial_{num}
            provider_dir = os.path.join(OUTPUT_DIR, provider, f"trial_{trial_num}")
            os.makedirs(provider_dir, exist_ok=True)
            
            for item in data:
                q_id_raw = str(item.get('id', 'unknown'))
                q_id = q_id_raw.zfill(2)
                sparql = item.get('generated_sparql')
                
                # Fallback if sparql is empty/null
                if not sparql:
                    sparql = f"# No SPARQL generated\n# Raw Response:\n{item.get('raw_llm_response', '')}"
                
                filename = f"q{q_id}.sparql"
                filepath = os.path.join(provider_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(sparql)
            
            print(f"Populated {provider} Trial {trial_num} with {len(data)} queries.")

    print(f"\nSPARQL files organized in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
