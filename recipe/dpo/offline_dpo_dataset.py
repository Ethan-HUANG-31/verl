import json
from torch.utils.data import Dataset
from typing import List, Optional

class OfflineDPODataset(Dataset):
    """
    支持 jsonl 偏好对格式的数据集，输出正负样本对，兼容 SPIN/DPO 训练流程。
    每条数据格式：
    {
        "problem": ..., "answer": ..., ...,
        "response": {"...", "choices": [{"text": ...}, ...]},
        "compressed_response": {"...", "choices": [{"text": ...}, ...]}
    }
    其中 response 作为 rejected, compressed_response 作为 chosen。
    """
    def __init__(self, data_files: List[str], tokenizer, processor=None, config=None):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.samples = []
        for file in data_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    js = json.loads(line)
                    # 只保留有正负样本的
                    if 'response' in js and 'compressed_response' in js:
                        self.samples.append(js)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        js = self.samples[idx]
        prompt = js.get('problem', '')
        # 正样本
        chosen = js['compressed_response']['choices'][0]['text']
        # 负样本
        rejected = js['response']['choices'][0]['text']
        # 拼接 prompt + response
        chosen_text = prompt + '\n' + chosen
        rejected_text = prompt + '\n' + rejected
        # Tokenize
        chosen_enc = self.tokenizer(chosen_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.config.data.max_prompt_length)
        rejected_enc = self.tokenizer(rejected_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.config.data.max_prompt_length)
        return {
            'prompt': prompt,
            'chosen_text': chosen_text,
            'rejected_text': rejected_text,
            'chosen_input_ids': chosen_enc['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_enc['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(0),
        }