# %% [markdown]
# ## Notebook for processesing CURIE-benchmark tasks using the **Cohere Command-R Plus model**.
# 

# %%
# @title Import Required Libraries
import os
import json
import pandas as pd
import numpy as np
# import altair as alt
import logging
import textwrap as tr
import torch
# from google.colab import drive
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
import concurrent.futures


# %%
# # @title Install and import Cohere
# ! pip install -U cohere
# import cohere

# %%
# # @title API Configuration
API_KEY = "YOUR_API_KEY"
MODEL_PATH = 'command-r-plus'
# MODEL_NAME = "doubao-1.5-thinking-pro"
file_root_path= os.path.join(os.path.dirname(__file__), "..")

# %%
# @title Mount Google Drive
# drive.mount('/content/drive', force_remount=True)
# os.chdir("/content/drive/My Drive")

# %%
# @title Configuration Classes
@dataclass
class ExperimentConfig:
    """Configuration class for experiment settings"""
    name: str
    base_dir: str
    inference_dir: str
    prompt_path: str

class ExperimentType(Enum):
    """Enum for different types of experiments"""
    PDB = "pdb"
    MPVE = "mpve"
    HFE = "hfe"
    GEO = "geo"
    DFT = "dft"
    HFD = "hfd"
    QECC_PDF = "qecc_pdf"
    QECC_TEX = "qecc_tex"

# %%
# @title Experiment Manager Class
class ExperimentManager:
    """Manages different experiment configurations"""
    def __init__(self, base_path: str = file_root_path):
        self.base_path = base_path
        self.experiments = self._initialize_experiments()

    def _initialize_experiments(self) -> Dict[ExperimentType, ExperimentConfig]:
        """Initialize all experiment configurations"""
        benchmark_path = f"{self.base_path}"
        return {
            ExperimentType.PDB: ExperimentConfig(
                name="PDB",
                base_dir=f"{self.base_path}/pdb",
                inference_dir=f"{self.base_path}/inference/multi_runs/current/pdb_new/reconstruct_protein_amino_acid_sequence_0_shot/",
                prompt_path=f"{benchmark_path}/prompts/reconstruct_protein_amino_acid_sequence_0_shot.txt"
            ),
            ExperimentType.MPVE: ExperimentConfig(
                name="MPVE",
                base_dir=f"{benchmark_path}/data/mpve",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/mpve/mat_paper_to_property_1_shot_exclude_trivia/",
                prompt_path=f"{benchmark_path}/prompts/mat_paper_to_property_1_shot_exclude_trivia.txt"
            ),
            ExperimentType.HFE: ExperimentConfig(
                name="HFE",
                base_dir=f"{benchmark_path}/data/hfe",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/hfe/extract_hamiltonian_0_shot/",
                prompt_path=f"{benchmark_path}/prompts/extract_hamiltonian_0_shot.txt"
            ),
            ExperimentType.GEO: ExperimentConfig(
                name="GEO",
                base_dir=f"{benchmark_path}/data/geo",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/geo/extract_dataset_from_geo_papers_0_shot",
                prompt_path=f"{benchmark_path}/prompts/extract_dataset_from_geo_papers_0_shot.txt"
            ),
            ExperimentType.DFT: ExperimentConfig(
                name="DFT",
                base_dir=f"{benchmark_path}/data/dft",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/dft/extract_dft_metadata_1_shot/",
                prompt_path=f"{benchmark_path}/prompts/extract_dft_metadata_1_shot.txt"
            ),
            ExperimentType.HFD: ExperimentConfig(
                name="HFD",
                base_dir=f"{benchmark_path}/data/hfd",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/hfd/derivation_prompt/",
                prompt_path=f"{benchmark_path}/prompts/derivation_prompt.txt"
            ),
            ExperimentType.QECC_PDF: ExperimentConfig(
                name="QECC_PDF",
                base_dir=f"{benchmark_path}/data/qecc_pdf",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/qecc_pdf/describe_code_in_paper/",
                prompt_path=f"{benchmark_path}/prompts/describe_code_in_paper.txt"
            ),
            ExperimentType.QECC_TEX: ExperimentConfig(
                name="QECC_TEX",
                base_dir=f"{benchmark_path}/data/qecc_tex",
                inference_dir=f"{benchmark_path}/inference/multi_runs/current/qecc_tex/describe_code_in_paper/",
                prompt_path=f"{benchmark_path}/prompts/describe_code_in_paper.txt"
            )
        }

    def get_config(self, experiment_type: ExperimentType) -> ExperimentConfig:
        """Get configuration for specific experiment type"""
        return self.experiments[experiment_type]

# %%
# @title Paper Processing Utilities
def specialize_prompt(template: str, tag: str, infil: str) -> str:
    """Replace a tag in a template with provided text."""
    if tag in template:
        return template.replace(tag, infil)
    raise ValueError(f'{tag} absent in template.')

def prepare_task_for_paper(paper: str, config: ExperimentConfig, model_id: str) -> dict:
    """Prepare the task information for a given paper."""
    paper_input = os.path.join(config.base_dir, 'inputs', f'{paper}.json')
    paper_gt = os.path.join(config.base_dir, 'ground_truth', f'{paper}.json')

    with open(paper_input, 'r') as f:
        inputs = json.load(f)
    with open(paper_gt, 'r') as f:
        targets = json.load(f)

    with open(config.prompt_path, 'r') as f:
        ptemp = f.read()

    spec_prompt = specialize_prompt(ptemp, '{{text}}', infil=inputs['text'])

    return {
        'record_id': paper,
        'model_id': MODEL_NAME,
        'prompt_path': config.prompt_path,
        'prompt_text': spec_prompt,
        
        'reasoning_content':'',
        'response_text': ''
    }

# %%
import time
import yaml
import requests
import traceback
def call_huoshan(messages, model_name="doubao-1.5-thinking-pro", config_path="./colabs/api_config.yaml"):
        """
        调用豆包模型接口，支持从配置文件中读取全部参数和带重试机制。
        import time
        import yaml
        import requests
        """
        # 加载模型配置
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                api_config = yaml.safe_load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file {config_path} not found.") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file: {e}") from e
        model_cfg = api_config.get(model_name)
        if not model_cfg:
            raise ValueError(f"模型 '{model_name}' 的配置信息未找到。")

        # 提取配置参数
        url = model_cfg.get("base_url")
        key = model_cfg.get("api_key")
        model = model_cfg.get("model_name")
        temperature = model_cfg.get("temperature", 0.2)
        top_p = model_cfg.get("top_p", 0.95)
        max_tokens = model_cfg.get("max_tokens", 4096)
        max_retries = model_cfg.get("max_retries", 3)
        retry_delay = model_cfg.get("retry_delay", 1.0)

        # 重试逻辑
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                data_json = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens
                    }
                response = requests.post(
                    url=url,
                    json=data_json,
                    headers={
                    "Authorization": f"Bearer {key}",
                    "x-ark-moderation-scene": "skip-ark-moderation"
                })
                response.raise_for_status()  # 捕捉非 2xx 状态码
                response_json = response.json()

                choice = response_json["choices"][0]
                finish_reason = choice["finish_reason"]
                reasoning_content = choice["message"].get("reasoning_content", None)
                content = choice["message"].get("content", None)

                if finish_reason == "stop":
                    if reasoning_content:
                        formatted_content = f"<think>\n{reasoning_content.strip()}\n</think>\n\n{content.strip()}"
                    else:
                        formatted_content = content.strip()
                else:
                    formatted_content = None

                return reasoning_content, content

            except Exception as e:
                traceback.print_exc()
                if attempt >= max_retries:
                    print(f"[Warning] get_llm_result_r1_full failed after {max_retries} attempts: {e}")
                    return None
                print(f"第 {attempt} 次调用失败：{e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避

# %%
# @title Paper Processor Class
class PaperProcessor:
    """Handles the processing of scientific papers"""

    def __init__(self, api_key: str, model_path: str):
        # self.co_v2 = cohere.ClientV2(api_key=api_key)
        self.model_path = model_path
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            filename='experiment_log.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_api_call(self, messages: List[Dict]) -> str:
        """Make API call with retry logic"""
        response = self.co_v2.chat(
            model=self.model_path,
            messages=messages,
            temperature=0.9,
            k=50,
            p=0.95,
            max_tokens=4000
        )
        return self._extract_response_text(response)

    def _extract_response_text(self, response) -> str:
        """Extract text from API response"""
        if hasattr(response, 'message'):
            if hasattr(response.message, 'content'):
                if isinstance(response.message.content, list):
                    return ' '.join(item.text for item in response.message.content if hasattr(item, 'text'))
                elif isinstance(response.message.content, str):
                    return response.message.content
        return str(response)

    def _save_result(self, task_info: dict, inference_dir: str, run_id: int, success: bool = True):
        """Save processing results"""
        status = 'success' if success else 'failure'
        output_dir = os.path.join(inference_dir, MODEL_NAME, f'run_{run_id}', status)
        os.makedirs(output_dir, exist_ok=True)

        serializable_task_info = {
            'record_id': task_info['record_id'],
            'model_id': task_info['model_id'],
            'prompt_path': task_info['prompt_path'],
            'reasoning_content': task_info['reasoning_content'],
            'prompt_text': task_info['prompt_text'],
            'response_text': str(task_info['response_text'])
        }

        output_file = os.path.join(output_dir, f'{task_info["record_id"]}.json')
        with open(output_file, 'w') as f:
            json.dump(serializable_task_info, f, indent=4,ensure_ascii=False)

    def process_papers(self, config: ExperimentConfig, run_range: range = range(1, 2)):
        """Process papers for given experiment configuration"""
        input_dir = os.path.join(config.base_dir, 'inputs')
        papers = [f.replace('.json', '') for f in os.listdir(input_dir) if f.endswith('.json')]

        self.logger.info(f"Starting processing {len(papers)} papers for {config.name}")

        for run_id in run_range:
            self.logger.info(f"Starting run {run_id + 1}")
            # for i, paper in enumerate(papers, 1):
            #     self.logger.info(f"Processing paper {i}/{len(papers)} in run {run_id + 1}")
            #     self._process_single_paper(paper, config, run_id)
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                # 提交所有任务
                future_to_paper = {
                    executor.submit(self._process_single_paper, paper, config, run_id): paper
                    for paper in papers
                }

                processed_count = 0
                for future in concurrent.futures.as_completed(future_to_paper):
                    paper_name = future_to_paper[future]
                    processed_count += 1
                    try:
                        # 获取结果，如果任务中发生未捕获的异常，这里会重新抛出
                        # _process_single_paper 内部已经有异常处理和日志记录，所以这里通常不会抛出
                        future.result()
                        # 你可以在这里记录成功完成，但 _process_single_paper 内部的日志可能已经足够
                        self.logger.info(f"Completed processing for paper {paper_name} ({processed_count}/{len(papers)}) in run {run_id + 1}.")
                    except Exception as exc:
                        # 这个异常通常不应该发生，因为 _process_single_paper 内部会捕获
                        self.logger.error(f"An unexpected error occurred for paper {paper_name} in run {run_id + 1} during future processing: {exc}")
            
            self.logger.info(f"Finished run {run_id + 1}.")

        self.logger.info(f"All runs completed for {config.name}.")

    def _process_single_paper(self, paper: str, config: ExperimentConfig, run_id: int):
        """Process a single paper"""
        try:
            task_info = prepare_task_for_paper(paper, config, self.model_path)

            if len(task_info['prompt_text'].split()) > 128000:
                raise ValueError("Input text exceeds token limit")

            # response = self._make_api_call([{
            #     "role": "user",
            #     "content": task_info['prompt_text']
            # }])
            # print(f"Processing paper {paper} ")
            # task_info['prompt_text'] = '你是谁' # 4000 tokens
            reasoning_content, response = call_huoshan([{
                "role": "user",
                "content": task_info['prompt_text']
            }], model_name=MODEL_NAME)
            print(f"paper {paper} finished, response: {response[:100]}...",end='')
            task_info['record_id'] = paper
            task_info['reasoning_content'] = reasoning_content
        
            task_info['response_text'] = response
            self._save_result(task_info, config.inference_dir, run_id, success=True)
            # time.sleep(2)  # Rate limiting

        except Exception as e:
            self.logger.error(f"Error processing paper {paper}: {str(e)}")
            task_info['response_text'] = str(e)
            self._save_result(task_info, config.inference_dir, run_id, success=False)
            time.sleep(2)

# %%
# @title Main Execution
def infer(task_name, infer_model_name):
    """Main execution function"""
    experiment_manager = ExperimentManager()

    processor = PaperProcessor(
        api_key=API_KEY,
        model_path=MODEL_PATH
    )
    MODEL_NAME = infer_model_name
    # Select experiment type
    if task_name == "MPVE":
        experiment_type = ExperimentType.MPVE
    elif task_name == "DFT":
        experiment_type = ExperimentType.DFT
    # experiment_type = ExperimentType.MPVE  # CHANGE THIS to process different experiments
    config = experiment_manager.get_config(experiment_type)

    processor.process_papers(config)

model_list=[
    "deepseek-r1",
    "doubao-1.5-thinking-pro",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1-Distill-Qwen-7B",
    
    
]
task_list=[
    "MPVE",
    "DFT" 
]

for task_name in task_list:
  # task_name= "MPVE"
  for infer_model_name in model_list:
    # run_evaluation("MPVE", infer_model_name)
    infer(task_name, infer_model_name)


