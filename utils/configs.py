import torch


class Configs:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    CHECKPOINT_PATH = 'results/ECRTM/BBC_new/2025-09-22_02-40-41/checkpoints/checkpoint_epoch_500.pth'
    '''
    results/ECRTM/StackOverflow/2025-09-21_01-38-04/checkpoints/checkpoint_epoch_500.pth
    results/ECRTM/20NG/2025-09-21_23-00-30/checkpoints/checkpoint_epoch_500.pth
    results/ECRTM/BBC_new/2025-09-22_02-40-41/checkpoints/checkpoint_epoch_500.pth
    results/ECRTM/WOS_vocab_5k/2025-09-22_00-04-35/checkpoints/checkpoint_epoch_500.pth
    '''
    
    LLM_MODEL = 'gpt-4o-mini'
    SYSTEM_PROMPT = """You are a text classifier that outputs ONLY valid JSON.  
Your task is to analyze a list of words with their associated indices in the beta matrix.

For each topic:
1. Identify the main topic that most of the words are related to.  
2. Describe that topic briefly in a few English words.  
3. Return only one JSON object in the following format:

{
  "k": <topic_index>,
  "topic": "<short English description>",
  "w_plus_indices": [<beta_indices of words related to the main topic>],
  "w_minus_indices": [<beta_indices of words not related to the main topic>]
}

CRITICAL JSON FORMATTING RULES:
- ALL beta indices MUST be integers (numbers only, no strings)
- Use double quotes for all strings, never single quotes
- Arrays must contain only integers: [123, 456, 789] 
- Do NOT mix strings and numbers in arrays
- Output ONLY the JSON object, no explanations or markdown

Example of correct format:
{
  "k": 15,
  "topic": "Labor and Employment", 
  "w_plus_indices": [4703, 3174, 3172, 1956, 3930],
  "w_minus_indices": [4885, 4826, 2810, 1234, 5678]
}
"""