import re

def get_option_label(index):
    return chr(65 + index)

def format_MathMC_options(data):
    return "\n".join([f"{opt['bullet']}: {opt['text']}" for opt in data])

def format_WildSci_options(data):
    return "\n".join([f"{k}: {v}" for k, v in data.items()])

def format_ScienceQA_options(data):
    formatted = []
    for i, text in enumerate(data):
        label = get_option_label(i)
        formatted.append(f"{label}. {text}")
    return "\n".join(formatted)

def format_ScienceQA_output(data):
    label = get_option_label(data)
    return label

def format_sciq_options(data):
    opts = []
    
    for i, opt in enumerate(data):
        label = get_option_label(i)
        opts.append(f"{label}. {opt}")
    return "\n".join(opts)

def format_pubmedqa_input(data):
    contexts = data.get("contexts", "")
    if isinstance(contexts, list):
        return "\n".join(contexts)
    return contexts

def format_natural_reasoning_finance_output(data):
    return data["response"]


def format_finance_slm_distillation_data_system(data):
    pattern = r'<\|im_start\|>system\n(.*?)\n<\|im_end\|>'
    match = re.search(pattern, data, re.DOTALL)

    if match:
        system_content = match.group(1)
        return system_content.strip()
    else:
        return ""

def format_huatuo_encyclopedia_qa_instruction(data):
    if isinstance(data, list):
        return data[0][0]
    return data

def format_huatuo_encyclopedia_qa_output(data):
    return data[0]

def format_canadian_tax_law_qa_instruction(data):
    inst_match = re.search(r'\[INST\](.*?)\[/INST\]', data, re.DOTALL)
    instruction = inst_match.group(1).strip() if inst_match else ""
    return instruction
    
def format_canadian_tax_law_qa_output(data):
    output_match = re.search(r'\[/INST\](.*?)</s>', data, re.DOTALL)
    output = output_match.group(1).strip() if output_match else ""
    return output
    
FORMAT_REGISTRY = {
    "format_MathMC_options": format_MathMC_options,
    "format_WildSci_options": format_WildSci_options,
    "format_ScienceQA_options": format_ScienceQA_options,
    "format_ScienceQA_output": format_ScienceQA_output,
    "format_sciq_options": format_sciq_options,
    "format_pubmedqa_input": format_pubmedqa_input,
    "format_natural_reasoning_finance_output": format_natural_reasoning_finance_output,
    "format_finance_slm_distillation_data_system": format_finance_slm_distillation_data_system,
    "format_huatuo_encyclopedia_qa_instruction": format_huatuo_encyclopedia_qa_instruction,
    "format_huatuo_encyclopedia_qa_output": format_huatuo_encyclopedia_qa_output,
    "format_canadian_tax_law_qa_instruction": format_canadian_tax_law_qa_instruction,
    "format_canadian_tax_law_qa_output": format_canadian_tax_law_qa_output,
}

def apply_formatters(example, config):
    if 'formatters' in config:
        for field, func_name in config['formatters'].items():
            if func_name in FORMAT_REGISTRY:
                example[field] = FORMAT_REGISTRY[func_name](example[field])
    return example

def filter_WildSci(example):
    r_ans = str(example.get('rationale_answer', '')).strip()
    ans = str(example.get('answer', '')).strip()
    
    return r_ans == ans

def filter_ScienceQA(example):
    return 'image' not in example or not example['image'] 


FILTER_REGISTRY = {
    "filter_WildSci": filter_WildSci,
    "filter_ScienceQA": filter_ScienceQA,
}

def apply_processing(example, config):
    if 'formatters' in config:
        for field, func_name in config['formatters'].items():
            if func_name in FORMAT_REGISTRY:
                example[field] = FORMAT_REGISTRY[func_name](example[field])
    
    if 'filters' in config:
        for filter_name in config['filters']:
            if filter_name in FILTER_REGISTRY:
                if not FILTER_REGISTRY[filter_name](example):
                    return None  
    
    return example