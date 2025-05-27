
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
import concurrent.futures
from model import call_huoshan,call_server
import logging



# # @title API Configuration
API_KEY = "YOUR_API_KEY"
MODEL_PATH = 'command-r-plus'
# MODEL_NAME = "deepseek-r1"
# "DeepSeek-R1-Distill-Qwen-7B"
# "DeepSeek-R1-Distill-Qwen-32B"
# MODEL_NAME = "deepseek-r1"
# "doubao-1.5-thinking-pro"

if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 命令行读取模型名、模型url、并发数，其中并发数为可选项
import sys
if len(sys.argv) > 1:
    model_name = sys.argv[1]  # 模型名
    model_url = sys.argv[2]  # 模型URL
    if len(sys.argv) > 3:
        concurrency = int(sys.argv[3])  # 可选的并发数
    else:
        concurrency = 32  # 默认并发数为1

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
    def __init__(self, base_path: str = ".."):
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

def prepare_task_for_paper(paper: str, config: ExperimentConfig, model_id: str,MODEL_NAME) -> dict:
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
# @title Paper Processor Class
class PaperProcessor:
    """Handles the processing of scientific papers"""

    def __init__(self, api_key: str, model_path: str,MODEL_NAME=None):
        # self.co_v2 = cohere.ClientV2(api_key=api_key)
        self.model_path = model_path
        self._setup_logging()
        self.MODEL_NAME=MODEL_NAME

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
        output_dir = os.path.join(inference_dir, self.MODEL_NAME, f'run_{run_id}', status)
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
            print(f"Starting run {run_id + 1}, using model:{self.MODEL_NAME}")
            # for i, paper in enumerate(papers, 1):
            #     self.logger.info(f"Processing paper {i}/{len(papers)} in run {run_id + 1}")
            #     self._process_single_paper(paper, config, run_id)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # 提交所有任务
                future_to_paper = {
                    executor.submit(self._process_single_paper, paper, config, run_id,self.MODEL_NAME): paper
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
                        # 在这里记录成功完成
                        self.logger.info(f"Completed processing for paper {paper_name} ({processed_count}/{len(papers)}) in run {run_id + 1}.")
                        print(f"Completed processing for paper {paper_name} ({processed_count}/{len(papers)}) in run {run_id + 1}.")
                    except Exception as exc:
                        # 这个异常通常不应该发生，因为 _process_single_paper 内部会捕获
                        self.logger.error(f"An unexpected error occurred for paper {paper_name} in run {run_id + 1} during future processing: {exc}")
            
            self.logger.info(f"Finished run {run_id + 1}.")

        self.logger.info(f"All runs completed for {config.name}.")

    def _process_single_paper(self, paper: str, config: ExperimentConfig, run_id: int,MODEL_NAME):
        """Process a single paper"""
        try:
            task_info = prepare_task_for_paper(paper, config, self.model_path,MODEL_NAME)

            if len(task_info['prompt_text'].split()) > 128000:
                raise ValueError("Input text exceeds token limit")

            # response = self._make_api_call([{
            #     "role": "user",
            #     "content": task_info['prompt_text']
            # }])
            # print(f"Processing paper {paper} ")
            # task_info['prompt_text'] = '你是谁' # 4000 tokens

            if MODEL_NAME == "doubao-1.5-thinking-pro" or MODEL_NAME == "deepseek-r1":
                reasoning_content, response = call_huoshan([{
                    "role": "user",
                    "content": task_info['prompt_text']
                }], model_name=MODEL_NAME)
            else:
                reasoning_content, response = call_server(messages=task_info['prompt_text'],
                                                          model_name=MODEL_NAME,
                                                          model_url=model_url
                                                          )
                
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
def infer(task_name, infer_model_name):
    """Main execution function"""
    MODEL_NAME = infer_model_name
    experiment_manager = ExperimentManager()
    
    processor = PaperProcessor(
        api_key=API_KEY,
        model_path=MODEL_PATH,
        MODEL_NAME= MODEL_NAME
    )
    
    # Select experiment type
    if task_name == "MPVE":
        experiment_type = ExperimentType.MPVE
    elif task_name == "DFT":
        experiment_type = ExperimentType.DFT
    # experiment_type = ExperimentType.MPVE  # CHANGE THIS to process different experiments
    config = experiment_manager.get_config(experiment_type)

    processor.process_papers(config,range(1, 2))

model_list=[
    "deepseek-r1",
    # "doubao-1.5-thinking-pro",
    # "DeepSeek-R1-Distill-Qwen-32B",
    # "DeepSeek-R1-Distill-Qwen-7B",
    # "chem_0320_phy_0324_2to1_math_ckpt_step624_ep2",
    # "chem_0320_phy_0324_2to1_math_add_r1_reasoning_ep1",
    # "chemistry_physics_math_7B_16k_rejection_sample_bs256_lr5e-6_roll16_on_aime_gpqa_scibench_global_step_50",
    # "our32b_s1math70w_code57w_liucong10w_ch_py_6k_32k"
     
]
task_list=[
    "MPVE",
    "DFT" 
]

# for task_name in task_list:
#   # task_name= "MPVE"
#   for infer_model_name in model_list:
#     # run_evaluation("MPVE", infer_model_name)
#     infer(task_name, infer_model_name)

# # # 使用threadPool调用不同模型
# for task_name in task_list:
  # task_name= "MPVE"
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(infer, task_name, model_name): task_name for task_name in task_list}
    for future in concurrent.futures.as_completed(futures):
        task_name = futures[future]
        try:
            future.result()
        except Exception as exc:
            print(f"{task_name} generated an exception: {exc}")
        else:
            print(f"{task_name} completed successfully.")


import curie_run_eval
for task_name in task_list:
    curie_run_eval.run_evaluation_signal_model(task_name,model_name,)


