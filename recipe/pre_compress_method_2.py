import os
import re
import traceback
import requests
import jsonlines
from tqdm import tqdm
import argparse
from multiprocessing import Pool

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='/remote-home/jiaanluo/work/AdaptThink/data/train/ref_presampling/DeepSeek-R1-Distill-Qwen-1.5B_aime_n0_K16_len16384_compressedthink_cleaned_nothink.json')
parser.add_argument("--output_path", type=str, default='./data/train/ref_presampling/DeepSeek-R1-Distill-Qwen-1.5B_aime_n0_K16_len16384_compressed_think_nothink.json')
parser.add_argument("--proc_num", type=int, default=512)
parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-1.5B")  
parser.add_argument("--max_tokens", type=int, default=16384)
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
model = args.model_name
max_tokens = args.max_tokens

# Define prompts
compress_prompt = "Please compress your answer to be more concise while preserving the key reasoning steps."
# 方法二的prompt
nothink_compress_prompt = "Please compress your answer to be more concise while preserving the key reasoning steps, and directly output the compressed reasoning process."

thinking_prompt = '<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>'
nothinking_prompt = '<｜begin▁of▁sentence｜><｜User｜>Use the reasoning below to directly answer the question.\n\n{question}\n\nReasoning: {short_thinking}<｜Assistant｜><think>\n</think>'

# 方法二第二轮输入的格式
nothinking_compress_prompt = '<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜><think>\n</think>'

# thinking_prompt = "{question}"
# nothink_prompt = "Use the reasoning below to directly answer the question without additional thinking or explanation.\n\n{question}\n\nReasoning: {short_thinking}"

def get_model_response(prompt, model_name, max_tokens=16384):
    response = requests.post(
        f'http://localhost:8000/v1/completions',
        headers={"Content-Type": "application/json"},
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "stream": False,
        },
        timeout=1200,
    ).json()
    return response

# Function to clean thinking tags from text
def clean_thinking_tags(text):
    # Remove <think>, </think>, <nothink>, </nothink> tags
    cleaned = re.sub(r'<\/?think>|<\/?nothink>', '', text)
    return cleaned

# Function to extract shorter thinking process from second round response
def extract_short_thinking(compressed_text):

    if compressed_text.count('</think>') == 1 and '<think>' not in compressed_text:
        thinking_text = compressed_text.split('</think>')[0]
        return thinking_text
    else:
        cleaned_text = clean_thinking_tags(compressed_text)
        paragraphs = cleaned_text.split("\n\n")
        if len(paragraphs) > 1:
            return "".join(paragraphs[:-1])
        return cleaned_text


# Check if the output file exists and which entries are already processed
processed_ids = set()
if os.path.exists(output_path):
    with jsonlines.open(output_path, 'r') as f:
        for js in tqdm(f):
            processed_ids.add(js['_id'])
    print(f"Found {len(processed_ids)} already processed examples")

# Load existing first-round responses
data = []
with jsonlines.open(input_path, 'r') as f:
    for js in tqdm(f):
        if js['_id'] not in processed_ids:
            data.append(js)

print(f"Loaded {len(data)} examples that need processing")

def process_example(js):
    try:
        question = js['problem'].strip()
        
        # ROUND 2: Get first response and clean thinking tags
        first_answer = js['response']['choices'][0]['text']
        cleaned_answer = clean_thinking_tags(first_answer)
        
        # Create compression question
        compression_question = question + "\n\n" + "Answer:" + cleaned_answer + "\n\n" + nothinking_compress_prompt
        
        # Use thinking prompt for compression
        compress_prompt_formatted = nothinking_compress_prompt.format(question=compression_question)

        # Get compressed response (Round 2)
        compressed_response = get_model_response(compress_prompt_formatted, model, max_tokens - 1)
        
        # Add compressed response to the data
        js['compressed_response'] = compressed_response
        
        # ROUND 3: Extract shorter thinking from the compressed response
        compressed_text = compressed_response['choices'][0]['text']

        short_thinking = extract_short_thinking(compressed_text)
        
        # Create third round question with nothink prompt
        third_round_question = nothinking_prompt.format(
            question=question,
            short_thinking=short_thinking
        )
        
        # Get final answer without thinking (Round 3)
        final_response = get_model_response(third_round_question, model, max_tokens - 1)
        
        # Add final response and extracted thinking to the data
        js['final_response'] = final_response
        # js['extracted_thinking'] = short_thinking
        
        # Write to file only after both rounds are complete
        with jsonlines.open(output_path, 'a') as f:
            f.write(js)
            
        return js
    except Exception as e:
        print(f"ERROR: {js['_id']}")
        traceback.print_exc()
        return None

# Process examples and write results to file
with Pool(args.proc_num) as p:
    results = list(tqdm(p.imap(process_example, data), total=len(data)))

successful = len([r for r in results if r is not None])
print(f"Completed processing {successful}/{len(data)} examples.")
print(f"Results saved to {output_path}")
