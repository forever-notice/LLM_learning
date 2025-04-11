import os
import time
import yaml
import openai
from typing import Dict, Any
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# Load environment variables and API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, 
                                 openai.error.RateLimitError, openai.error.ServiceUnavailableError, 
                                 openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    """
    Call OpenAI API with exponential backoff retry logic
    """
    return openai.ChatCompletion.create(**kwargs)

def read_labels(file_path: str) -> list:
    """
    Read action labels from file
    """
    with open(file_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    return labels

def generate_kg_for_action(label: str) -> Dict[str, Any]:
    """
    Generate knowledge graph for a single action label
    """
    prompt = f"""You are a commonsense knowledge base, especially for human actions. You will be provided with an action label below, which is delimited with square brackets. Use the following step-by-step instructions to respond to user inputs:

Conditions
1 - Return the object entity list contained Top K most relevant objects involved in the given action (0<=K<=3).
2 - What are the relations among these object entities? Find the proper predicate names that concisely describing the relationship between each object pair chosen from the object entity list.
3 - What are the relations between the given action entity and these object entities? Choose the proper predicate names that concisely describing the relationship between the given action entity and each object entity listed above.
4 - What sub-actions does the given action entity involve? The sub-action name should contain as less object name as possible. Return each sub-action name in the right processing order.
5 - Generate the action category info based on the instructions above in YAML format, reduce other prose.

Questions
Should include these fields: [label (i.e., the given action name), obj_li (i.e., object list), obj_rel_triples (i.e., object-object relation triples), act_obj_triples (i.e., action-object relation triples), sub_act_li (i.e., sub-action entity list), sub_act_rel_triples (i.e., subaction-subaction relation triples)], under the root "given action name". The triples should be in this format: <...,...,...>.

[{label}]
"""
    
    start_time = time.time()
    try:
        response = chat_completion_with_backoff(
            model="gpt-4",  # Can be adjusted based on needs
            messages=[
                {"role": "system", "content": "You are a commonsense knowledge base specialized in creating knowledge graphs for human actions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        content = response.choices[0]["message"]["content"].strip()
        
        # Try to parse YAML
        try:
            # Parse the returned YAML
            yaml_content = yaml.safe_load(content)
            return yaml_content
        except yaml.YAMLError:
            # If parsing fails, return raw content
            return {label: {"raw_content": content}}
            
    except Exception as e:
        print(f"Error generating KG for '{label}': {e}")
        return {label: {"error": str(e)}}
    finally:
        end_time = time.time()
        print(f"Time to process '{label}': {end_time - start_time:.3f} seconds")

def main():
    # File path configuration
    root_path = "/usr1/home/s124mdg53_04/Dissertation/ASKG_utils"  # Set your root directory path here
    input_file = os.path.join(root_path, "test.txt")
    output_file = os.path.join(root_path, "action_knowledge_graph.yaml")
    
    # Read labels
    print(f"Reading action labels from {input_file}...")
    labels = read_labels(input_file)
    print(f"Successfully read {len(labels)} action labels.")
    
    # Track progress
    total_labels = len(labels)
    all_kg_data = {}
    
    # Continue from last run (default is 0)
    last_processed = 0
    
    # Process labels and generate KG
    for i, label in enumerate(labels[last_processed:], start=last_processed+1):
        print(f"Processing [{i}/{total_labels}]: {label}")
        
        # Generate KG for this label
        kg_data = generate_kg_for_action(label)
        all_kg_data.update(kg_data)
        
        # Save intermediate results to avoid losing progress
        if i % 5 == 0:  # Save every 5 labels
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(all_kg_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            print(f"Intermediate results saved at {i}/{total_labels}")
    
    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(all_kg_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"Knowledge graph generation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()