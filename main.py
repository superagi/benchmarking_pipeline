import argparse
import json
import logging
import os
import yaml

from lm_eval import tasks, evaluator, utils
from accelerate import Accelerator
accelerator = Accelerator()

logging.getLogger("openai").setLevel(logging.WARNING)

def get_config_with_defaults(config, task_config_key, default_num_fewshot=0, default_batch_size=16, default_use_cache=True):
    task_config = config.get(task_config_key, {})
    num_fewshot = task_config.get('num_fewshot', default_num_fewshot)
    batch_size = task_config.get('batch_size', default_batch_size)
    use_cache = task_config.get('use_cached_result', default_use_cache)
    return num_fewshot, batch_size, use_cache

def main():
    print(f"Tasks to run: ['hellaswag', 'gsm8k', 'arc_challenge']")
    

    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    print("pretrained model: ", config['model'])
    device = accelerator.device
    
    model_args = 'pretrained=' + config['model']
    dirname = os.path.dirname('./results')


    num_fewshot_gsm8k, batch_size_gsm8k, use_cache_gsm8k = get_config_with_defaults(config, 'gsm8k_config')
    results_gsm8k = evaluator.simple_evaluate(
        model='hf-causal-experimental',
        model_args=model_args,
        tasks=['gsm8k'],
        num_fewshot=num_fewshot_gsm8k,
        batch_size=batch_size_gsm8k,
        device=device,
        no_cache= not use_cache_gsm8k,
    )
    dumped_results_gsm8k = json.dumps(results_gsm8k, indent=2)
    print("dumped_results_gsm8k: ", dumped_results_gsm8k)
    with open('./results', "w") as f:
        f.write(dumped_results_gsm8k)

    
    num_fewshot_arc, batch_size_arc, use_cache_arc = get_config_with_defaults(config, 'arc_challenge_config')
    results_arcchallenge = evaluator.simple_evaluate(
        model='hf-causal-experimental',
        model_args=model_args,
        tasks=['arc_challenge'],
        num_fewshot=num_fewshot_arc,
        batch_size=batch_size_arc,
        device=device,
        no_cache= not use_cache_arc,
    )
    dumped_results_arcchallenge = json.dumps(results_arcchallenge, indent=2)
    print("dumped_results_arcchallenge: ", dumped_results_arcchallenge)
    with open('./results', "a") as f:
        f.write(dumped_results_arcchallenge)

    
    num_fewshot_hella, batch_size_hella, use_cache_hella = get_config_with_defaults(config, 'hellaswag_config')
    results_hellaswag = evaluator.simple_evaluate(
        model='hf-causal-experimental',
        model_args=model_args,
        tasks=['hellaswag'],
        num_fewshot=num_fewshot_hella,
        batch_size=batch_size_hella,
        device=device,
        no_cache= not use_cache_hella,
    )
    dumped_results_hellaswag = json.dumps(results_hellaswag, indent=2)
    print("dumped_results_hellaswag: ", dumped_results_hellaswag)
    with open('./results', "a") as f:
        f.write(dumped_results_hellaswag)


    print(evaluator.make_table(dumped_results_arcchallenge))

if __name__ == "__main__":
    main()
