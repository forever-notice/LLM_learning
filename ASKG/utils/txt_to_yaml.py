#!/usr/bin/env python3
import sys
import yaml

def convert_txt_to_yaml(input_file, output_file):
    """Convert each line in txt file to a YAML item."""
    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create a list from the lines
    yaml_data = lines
    
    # Write to YAML file
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python txt_to_yaml.py input.txt output.yaml")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_txt_to_yaml(input_file, output_file)
    print(f"Conversion complete: {input_file} → {output_file}")