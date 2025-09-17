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
    parser.add_argument("--nothinking", action='store_true', default=False)
    parser.add_argument("--use_compressed", action='store_true', default=False,
                        help="Use compressed responses instead of original ones")
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
    
    # Choose which responses to use - original or compressed
    # if args.use_compressed and 'compressed_response' in items[0]:
    #     response_field = 'compressed_response'
    #     print("Using compressed responses")
    # else:
    #     response_field = 'response'
    #     print("Using original responses")
    if args.use_compressed and 'final_response' in items[0]:
        response_field = 'final_response'
        # print("Using compressed responses")
    else:
        response_field = 'response'
        print("Using original responses")
    
    # Check for truncated responses
    truncates = [(item[response_field]['choices'][0]['finish_reason'] == 'length') for item in items]
    
    # Get solutions with appropriate handling for thinking/nothinking
    solutions = [(('</think>' if args.nothinking else '') + item[response_field]['choices'][0]['text']) for item in items]

    # Calculate token lengths
    lengths = [item[response_field]['usage']['completion_tokens'] for item in items]
    if args.nothinking:
        lengths = [length + 1 for length in lengths]  # Add 1 if nothinking is True
    
    # Calculate correctness
    correctness = [adapt_think_rm(data_source='', solution_str=solution, ground_truth=real_answer)['acc'] for solution in solutions]
    
    avg_acc = np.mean(correctness)
    avg_len = np.mean(lengths)
    avg_clip_ratio = np.mean(truncates)
    
    # Sometimes tokenizer would change some special tokens
    processed_problem = tokenizer.decode(tokenizer.encode(problem, add_special_tokens=False))
    
    return {
        'problem': processed_problem,
        'answer': real_answer,
        # 'response_type': 'compressed' if args.use_compressed else 'original',
        'response_type': 'final' if args.use_compressed else 'original',
        'metrics': {
            'n_responses': len(items),
            'avg_acc_thinking': avg_acc,
            'avg_len_thinking': avg_len,
            'avg_clip_ratio': avg_clip_ratio,
        }
    }




# # 在下面添加这段代码，替换原有的多进程处理部分
# # 只处理第一个问题
# first_problem = list(data.keys())[0]
# print(f"只处理第一个问题: {first_problem}")
# result = process(first_problem)
# print("\n单条结果:")
# print(result)

# # 可以在这里添加任何其他调试输出
# print("\n详细信息:")
# print(f"问题: {result['problem']}")
# print(f"答案: {result['answer']}")
# print(f"指标: {result['metrics']}")

# exit(0)  # 处理完第一条数据后直接退出

with Pool(64) as p:
    results = list(tqdm(p.imap(process, list(data.keys())), total=len(data)))

overall_metrics = {}
for key in results[0]['metrics']:
    overall_metrics[key] = np.mean([js['metrics'][key] for js in results])

# response_type = 'compressed' if args.use_compressed else 'original'
response_type = 'final' if args.use_compressed else 'original'
results = [{'problem': '__OVERALL__', 'answer': None, 'response_type': response_type, 'metrics': overall_metrics}] + results

print(f"\nResults for {response_type} responses:")
for key, value in overall_metrics.items():
    print(f'{key}: {value}')

save_dir = args.output_path.rsplit('/', 1)[0]
os.makedirs(save_dir, exist_ok=True)
json.dump(results, open(args.output_path, 'w'), indent=2, ensure_ascii=False)
print(f"Results saved to {args.output_path}")
