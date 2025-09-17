import json, jsonlines
from tqdm import tqdm
from adapt_think_rm import adapt_think_rm
from multiprocessing import Pool
import os
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/remote-home/share/jiaanluo/adapt/mine/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--answer_key", type=str, default="answer")
    return parser.parse_args()

args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Group data by problem
data = defaultdict(list)
for js in tqdm(jsonlines.open(args.input_path, "r")):
    problem = js['problem'].strip()
    data[problem].append(js)

print("num data:", len(data))




def process(problem):
    items = data[problem]
    assert problem == items[0]['problem'].strip()
    real_answer = items[0][args.answer_key]
    
    response = "response"
    compressed_response = "compressed_response"
    final_response = "final_response"
    
    solutions_response = [item[response]['choices'][0]['text'] for item in items]
    solutions_compressed_response = [('</think>' + item[compressed_response]['choices'][0]['text']) for item in items]
    solutions_final_response = [('</think>' + item[final_response]['choices'][0]['text']) for item in items]
 
    # Calculate correctness
    correctness_response = [adapt_think_rm(data_source='', solution_str=solution, ground_truth=real_answer)['acc'] for solution in solutions_response]
    correctness_compressed_response = [adapt_think_rm(data_source='', solution_str=solution, ground_truth=real_answer)['acc'] for solution in solutions_compressed_response]
    correctness_final_response = [adapt_think_rm(data_source='', solution_str=solution, ground_truth=real_answer)['acc'] for solution in solutions_final_response]


    avg_acc_response = np.mean(correctness_response)
    avg_acc_compressed_response = np.mean(correctness_compressed_response)
    avg_acc_final_response = np.mean(correctness_final_response)

    # Sometimes tokenizer would change some special tokens
    processed_problem = tokenizer.decode(tokenizer.encode(problem, add_special_tokens=False))
    
    return {
        'problem': processed_problem,
        'answer': real_answer,
        'response': solutions_response,
        'compressed_response': solutions_compressed_response,
        'final_response': solutions_final_response,
        'metrics': {
            'n_responses': len(items),
            'acc_response': correctness_response,
            'acc_compressed_response': correctness_compressed_response,
            'acc_final_response': correctness_final_response,
            'avg_acc_response': avg_acc_response,
            'avg_acc_compressed_response': avg_acc_compressed_response,
            'avg_acc_final_response': avg_acc_final_response,
        }
    }


with Pool(64) as p:
    results = list(tqdm(p.imap(process, list(data.keys())), total=len(data)))




#-------------------------------------------------------------------------------------------------

# overall_metrics = {}
# for key in results[0]['metrics']:
#     overall_metrics[key] = np.mean([js['metrics'][key] for js in results])


#------------------------------------------------------------------------------------------------
# response_type = 'compressed' if args.use_compressed else 'original'






# overall_metrics = {}

# list_keys = {'acc_response', 'acc_compressed_response', 'acc_final_response'}

# per_problem_metrics = [js['metrics'] for js in results]  
# for key in results[0]['metrics'].keys():
#     if key in list_keys:
#         flat = [v for m in per_problem_metrics for v in m[key]]
#         overall_metrics[key] = float(np.mean(flat)) if len(flat) > 0 else 0.0

#     elif key == 'n_responses':
#         overall_metrics[key] = int(np.sum([m[key] for m in per_problem_metrics]))

#     elif key.startswith('avg_'):
#         vals = [m[key] for m in per_problem_metrics]
#         weights = [m['n_responses'] for m in per_problem_metrics]
#         overall_metrics[key] = float(np.average(vals, weights=weights)) if np.sum(weights) > 0 else 0.0

#     else:
#         overall_metrics[key] = float(np.mean([m[key] for m in per_problem_metrics]))




#-------------------------------------------------------------------------------------------------------------
# results = [{'problem': '__OVERALL__', 'answer': None, 'metrics': overall_metrics}] + results

# print(f"\nResults for:")
# for key, value in overall_metrics.items():
#     print(f'{key}: {value}')

save_dir = args.output_path.rsplit('/', 1)[0]
os.makedirs(save_dir, exist_ok=True)
json.dump(results, open(args.output_path, 'w'), indent=2, ensure_ascii=False)
print(f"Results saved to {args.output_path}")
