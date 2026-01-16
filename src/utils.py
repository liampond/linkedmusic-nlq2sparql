import os
import yaml
from dotenv import load_dotenv

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_ontology(ontology_path):
    """Loads the ontology file content."""
    if not os.path.exists(ontology_path):
        return ""
    with open(ontology_path, "r") as f:
        return f.read()

def setup_env():
    """Loads environment variables."""
    load_dotenv()
