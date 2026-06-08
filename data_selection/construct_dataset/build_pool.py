import yaml
import os
import json
import uuid
from datasets import load_dataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def standardize_format(example, domain, ds_name, mapping):
    instruction = example.get(mapping.get('instruction'), "请提供详细解答。")
    if 'instruction' not in mapping and 'default_instruction' in mapping:
        instruction = mapping['default_instruction']
        
    return {
        "sample_id": str(uuid.uuid4()),
        "source_dataset": ds_name,
        "domain": domain,
        "instruction": instruction,
        "input": example.get(mapping['input'], ""),
        "output": example.get(mapping['output'], ""),
        "metadata": {"original_features": list(example.keys())}
    }

def build_data_pools():
    config = load_config()
    domains = config.get('domains', {})
    global_size = config.get('global_sampling', {}).get('sizes', 1000)

    for domain_name, domain_cfg in domains.items():
        print(f"--- 正在处理领域: {domain_name} ---")
        domain_size = domain_cfg.get('sampling_size', global_size)
        for ds_cfg in domain_cfg.get('datasets', []):
            ds_name = ds_cfg['name']
            try:
                raw_ds = load_dataset(ds_name, split=ds_cfg['split'], streaming=True)
                sampled_data = raw_ds.take(domain_size)
                pool_data = []
                for item in sampled_data:
                    standardized = standardize_format(item, domain_name, ds_name, ds_cfg['mapping'])
                    pool_data.append(standardized)
                
                output_dir = os.path.join("data_pool", domain_name, str(domain_size))
                os.makedirs(output_dir, exist_ok=True)
                
                with open(os.path.join(output_dir, f"{ds_name.replace('/', '_')}.jsonl"), "w", encoding="utf-8") as f:
                    for entry in pool_data:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            
            except Exception as e:
                print(f"处理 {ds_name} 失败: {e}")

if __name__ == "__main__":
    build_data_pools()