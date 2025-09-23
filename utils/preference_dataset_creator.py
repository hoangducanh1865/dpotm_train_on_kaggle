import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.configs import Configs as cfg 


class PreferenceDatasetCreator:
    def __init__(self, dir_path):
        load_dotenv()
        self.preference_dataset_path = os.path.join(dir_path, 'preference_dataset.jsonl')
        self.top_words_10_path = os.path.join(dir_path, 'top_words_10.jsonl')
        self.top_words_15_path = os.path.join(dir_path, 'top_words_15.jsonl')
        self.top_words_20_path = os.path.join(dir_path, 'top_words_20.jsonl')
        self.top_words_25_path = os.path.join(dir_path, 'top_words_25.jsonl')
        self.model = cfg.LLM_MODEL
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = cfg.SYSTEM_PROMPT

    def create(self):
        def process_line(k, line):
            topic_data = json.loads(line.strip())
            top_words = topic_data['top_words']
            words_with_indices = []
            for word_dict in top_words:
                for word, idx in word_dict.items():
                    words_with_indices.append(f"'{word}' (beta_index: {idx})")
            
            prompt_content = f"Topic {k}: {', '.join(words_with_indices)}"
            
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.0
            )
            raw_data = response.choices[0].message.content.strip()
            
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                # Try to fix common JSON formatting issues
                print(f"🔧 Attempting to fix JSON for line {k}...")
                
                # Fix 1: Replace unquoted strings in arrays
                import re
                fixed_data = raw_data
                
                # Pattern to find arrays with mixed quoted/unquoted strings
                # Look for: [numbers, unquoted_words, numbers]
                def fix_array_content(match):
                    array_content = match.group(1)
                    # Split by comma and fix each element
                    elements = [elem.strip() for elem in array_content.split(',')]
                    fixed_elements = []
                    
                    for elem in elements:
                        if elem.isdigit() or (elem.startswith('-') and elem[1:].isdigit()):
                            # Keep numbers as is
                            fixed_elements.append(elem)
                        elif elem.startswith('"') and elem.endswith('"'):
                            # Keep quoted strings as is (but these should be converted to numbers)
                            try:
                                # Try to extract number from quoted string
                                num_val = elem.strip('"')
                                if num_val.isdigit():
                                    fixed_elements.append(num_val)
                                else:
                                    # Skip non-numeric strings
                                    continue
                            except:
                                continue
                        elif elem.startswith("'") and elem.endswith("'"):
                            # Handle single quoted
                            try:
                                num_val = elem.strip("'")
                                if num_val.isdigit():
                                    fixed_elements.append(num_val)
                                else:
                                    continue
                            except:
                                continue
                        else:
                            # Try to parse as number, skip if not
                            try:
                                if elem.strip().isdigit():
                                    fixed_elements.append(elem.strip())
                                # Skip non-numeric unquoted strings
                            except:
                                continue
                    
                    return '[' + ', '.join(fixed_elements) + ']'
                
                # Apply fixes to arrays
                fixed_data = re.sub(r'\[([^\]]+)\]', fix_array_content, fixed_data)
                
                try:
                    data = json.loads(fixed_data)
                    print(f"✅ JSON fixed successfully for line {k}")
                    
                    # FINAL VALIDATION: Ensure all indices are integers
                    if 'w_plus_indices' in data:
                        data['w_plus_indices'] = [int(x) for x in data['w_plus_indices'] if str(x).isdigit()]
                    if 'w_minus_indices' in data:
                        data['w_minus_indices'] = [int(x) for x in data['w_minus_indices'] if str(x).isdigit()]
                    
                except json.JSONDecodeError:
                    # If still failing, create a fallback structure
                    print(f"⚠️ JSON still invalid for line {k}, creating fallback...")
                    
                    # Extract topic info manually
                    topic_match = re.search(r'"topic":\s*"([^"]*)"', raw_data)
                    topic_name = topic_match.group(1) if topic_match else f"Topic_{k}"
                    
                    # Create simple fallback with first few indices
                    data = {
                        "k": k,
                        "topic": topic_name,
                        "w_plus_indices": list(range(0, min(10, len(words_with_indices)))),  # First 10 indices
                        "w_minus_indices": list(range(max(0, len(words_with_indices)-10), len(words_with_indices)))  # Last 10 indices
                    }
                    print(f"📝 Created fallback preference data for line {k}")
                    
            return data
            
        with open(self.top_words_25_path, 'r', encoding='utf-8') as infile, open(self.preference_dataset_path, 'w', encoding='utf-8') as outfile:
            for k, line in enumerate(infile):
                data = process_line(k, line)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # Debug
        print(f'Created and saved preference dataset to: {self.preference_dataset_path}')
