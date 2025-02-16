import json
import torch
import numpy as np
import random
import copy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def sample_and_dump_incomplete_data(ratio, data):
    result = []
    for sample in data:
        temp_data = copy.deepcopy(sample)
        temp_data['labels'] = random.sample(sample['labels'], int(np.ceil(len(sample['labels']) * ratio) ) )
        result.append(temp_data)
    with open(f"TEST_train_incomplete_{ratio}.json", "w") as fh:
        json.dump(result, fh)

if __name__ == "__main__":
    full_train_data = json.load(open('./train_annotated.json', 'r'))
    sample_and_dump_incomplete_data(0.2, full_train_data)
    sample_and_dump_incomplete_data(0.4, full_train_data)
    sample_and_dump_incomplete_data(0.6, full_train_data)
    sample_and_dump_incomplete_data(0.8, full_train_data)