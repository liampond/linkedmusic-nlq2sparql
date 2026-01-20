import os
import glob
import re

def combine_ontologies(ontology_dir="ontologies", output_file="ontologies/combined.ttl", ignore_files=None):
    """
    Combines multiple Turtle (.ttl) ontology files into a single file.
    Extracts prefixes to the top and concatenates the rest of the content.
    """
    if ignore_files is None:
        ignore_files = ["ontology_Aug2025.ttl", "combined.ttl", "cantusdb.ttl", "rism.ttl", "cantusindex.ttl"]

    ttl_files = glob.glob(os.path.join(ontology_dir, "*.ttl"))
    
    # Use a dict to store prefix -> url mapping for true deduplication
    prefix_map = {}
    content_lines = []
    
    # Regex to parse prefix lines: @prefix name: <url> .
    # Handles variable whitespace
    prefix_pattern = re.compile(r'@prefix\s+([\w-]+):\s+<([^>]+)>\s*\.')

    print(f"Found {len(ttl_files)} ontology files in {ontology_dir}")

    # Process files
    # Sort files to ensure deterministic output order
    for file_path in sorted(ttl_files):
        filename = os.path.basename(file_path)
        if filename in ignore_files:
            continue
            
        print(f"Processing {filename}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                # 1. Clean up trailing newlines from previous content to ensure clean state
                while content_lines and content_lines[-1].strip() == "":
                    content_lines.pop()
                
                # 2. Add separation from previous content (if any)
                if content_lines:
                    # Ensure the last line ends with a newline before adding separator
                    if not content_lines[-1].endswith("\n"):
                        content_lines[-1] += "\n"
                    content_lines.append("\n") # One empty line separator
                
                # 3. Add Header
                content_lines.append(f"# --- Source: {filename} ---\n")
                
                # 4. Add separation after header
                content_lines.append("\n") # One empty line separator
                
                start_doc = True
                for line in lines:
                    stripped = line.strip()
                    match = prefix_pattern.match(stripped)
                    if match:
                        p_name = match.group(1)
                        p_url = match.group(2)
                        prefix_map[p_name] = p_url
                    elif stripped.lower().startswith("@prefix"):
                         print(f"Warning: Line looked like prefix but failed regex: {stripped}")
                    elif stripped == "":
                         if start_doc:
                             continue # Skip leading blank lines in the file
                         
                         # Check if the previous line was also empty to avoid double newlines within files
                         if content_lines and content_lines[-1].strip() == "":
                             pass
                         else:
                             content_lines.append(line)
                    else:
                        start_doc = False # Found content
                        content_lines.append(line)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Write combined output
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # Write prefixes first
            f.write("# --- Prefixes ---\n")
            for p_name in sorted(prefix_map.keys()):
                p_url = prefix_map[p_name]
                f.write(f"@prefix {p_name}:\t<{p_url}> .\n")
            
            f.write("\n# --- Ontology Definitions ---\n\n")
            
            # Ensure we don't start with a blank line if we don't want to
            # But the logic above might leave one leading newline if content_lines started empty?
            # actually logic: `if content_lines: append('\n')` protects the very start.
            # But the first file loop does: `if content_lines:` (false) -> `append header`.
            # So the first file header starts immediately.
            
            for line in content_lines:
                f.write(line)

            # Since content_lines preserves newlines, we check the last one
            if content_lines and not content_lines[-1].endswith('\n'):
                 f.write('\n') # finish the last line if needed
            
            f.write('\nREMEMBER: Please find the correct QIDs')
                
        print(f"Successfully created combined ontology at: {output_file}")
        return output_file
    except Exception as e:
        print(f"Failed to write output file: {e}")
        return None

if __name__ == "__main__":
    combine_ontologies()
