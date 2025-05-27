
import json5
import re
from typing import Any, Tuple, Union, Dict
import ast
import os
import json_repair
# import google.generativeai as genai
# import Levenshtein
# MODEL_NAME = "deepseek-r1"
# "doubao-1.5-thinking-pro"
# EVAL_MODEL_NAME = "deepseek-r1"   # eval by openai only

# TASK_NAME = "MPVE"
# "DFT"
# "deepseek-r1"
# INFER_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B"
# # "DeepSeek-R1-Distill-Qwen-7B"
# # "DeepSeek-R1-Distill-Qwen-32B"
# # MODEL_NAME = "deepseek-r1"
# # "doubao-1.5-thinking-pro"

model_list=[
    # "deepseek-r1",
    # "doubao-1.5-thinking-pro",
    # "DeepSeek-R1-Distill-Qwen-32B",
    # "DeepSeek-R1-Distill-Qwen-7B",
    "chem_0320_phy_0324_2to1_math_ckpt_step624_ep2",
    "chem_0320_phy_0324_2to1_math_add_r1_reasoning_ep1",
    "chemistry_physics_math_7B_16k_rejection_sample_bs256_lr5e-6_roll16_on_aime_gpqa_scibench_global_step_50",
    "our32b_s1math70w_code57w_liucong10w_ch_py_6k_32k",
]
task_list=[
    "MPVE",
    "DFT" 
]

# # --- 从命令行参数获取模型名称 ---
# if "__file__" in globals() :
#     # 表示当前是一个py文件
#     if len(sys.argv) > 1:
#         INFER_MODEL_NAME = sys.argv[1]
#         print(f"从命令行参数获取模型名称: {INFER_MODEL_NAME}")
#     else:
#         # 如果没有提供命令行参数，使用默认值或提示错误
#         INFER_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B" # 默认值
#         print(f"警告: 未在命令行提供模型名称，将使用默认值: {INFER_MODEL_NAME}")




if "__file__" in globals() :
    # 表示当前是一个py文件
    os.chdir(os.path.dirname(__file__))
    print("当前路径:", os.getcwd())
    print(os.path.abspath('./'))
file_root_path=".."

# %% [markdown]
# # eval functions

# %% [markdown]
# ### 服务器模型

# %%
from model import *
# call_server(messages="你是谁")

# %%
# 读取配置文件
if "__file__" in globals() :
    # 表示当前是一个py文件
    config_file = os.path.join(os.path.dirname(__file__), "api_config.yaml")
else:
    # 表示当前是一个交互式环境
    config_file = os.path.join(os.getcwd(), "api_config.yaml")
# config_file = "api_config.yaml"
api_config = yaml.safe_load(open(config_file, "r", encoding="utf-8"))
model_name = "gpt-4o-2024-11-20"
model_cfg = api_config.get(model_name)
from openai import OpenAI
openai_client = OpenAI(
        api_key= api_config.get(model_name, {}).get("api_key"),
        base_url= api_config.get(model_name, {}).get("base_url"),
        max_retries= api_config.get(model_name, {}).get("max_retries"),
        timeout= api_config.get(model_name, {}).get("timeout"),
    )

# %% [markdown]
# ### LLMSim util

# %%

def eval_overall_result(
    eval_output_item: dict[str, Any], verbose: bool = False
) -> dict[str, Any]:
  """Gets overall eval result.

  Args:
    eval_output_item: eval output item.
    verbose: whether to print model eval original output.

  Returns:
    overall eval result.
  """
  num_match = sum([
      1 if ("json_extracted_index" in item) else 0
      for item in eval_output_item["response_json"]
  ])
  if verbose:
    print("Model eval original output:\n", eval_output_item)
  num_gt = eval_output_item["ground_truth_length"]
  num_response = eval_output_item["model_response_length"]
  pre = min(num_match / num_response if num_response else 0, 1.0)
  rec = min(num_match / num_gt if num_gt else 0, 1.0)
  return {
      "num_match": num_match,
      "num_ground_truth": num_gt,
      "num_model_response": num_response,
      "precision": pre,
      "recall": rec,
      "f1": 2.0 * pre * rec / (pre + rec) if pre + rec else 0.0,
  }

# def llm_output(client: Any, prompt: str) -> str:
#   # client=None for external api
#   return get_model().generate_content(prompt).text
def load_matsci_prompt(filepath: str) -> str:
  """Loads matsci prompt.

  Args:
    filepath: filepath of prompt.

  Returns:
    Loaded prompt.
  """
  # return resources.GetResource(filepath).decode("utf-8").strip()
  with open(filepath, 'r') as file:
    text_content = file.read()
  return text_content.strip()


def model_eval_json(
    record_id: str | None,
    json_ground_truth: list[dict[str, Any]],
    json_model_response: list[dict[str, Any]],
    eval_prompt: str,
    client: Any,
) -> dict[str, Any]:
  """Evaluate with json ground truth and model response."""
  eval_list = []
  # genai.configure(api_key=userdata.get('GEMINI_API_KEY'))
  
  for j, ground_truth_item in enumerate(json_ground_truth):
    
    for k, model_response_item in enumerate(json_model_response):
      # Add index to llm input to suppress hallucination on json_extracted_index
      # prediction
      try:
        model_response_item["json_extracted_index"] = k
      except Exception as e:
        print("Error in loading prompt: ", e)
        print("record_id: ", record_id)
        # print("json_ground_truth: ", json_ground_truth)
        print("model_response_item: ", model_response_item)
    prompt = (
        load_matsci_prompt(eval_prompt)
        .replace(
            "{{json_ground_truth}}", json5.dumps(ground_truth_item, indent=2)
        )
        .replace(
            "{{json_extracted_list}}", json5.dumps(json_model_response, indent=2)
        )
    )
  
    _ , output = call_openai(
      messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
      client=openai_client
    )


    try:
      output_json = json_repair.repair_json(output,return_objects=True)
    except Exception as e:  # pylint: disable=broad-except
      print("Skipping incomplete last item in output: ", e)
      inds = [m.start() for m in re.finditer(r",\s*\{", output)]
      if inds:
        ind = inds[-1]
        output_json = json5.loads(output[:ind] + "]")
      else:
        output_json = []
    if isinstance(output_json, list):
      # Handle edge case that model hallucinated outputing list enclosing json.
      if not output_json:
        output_json = {}
      else:
        output_json = output_json[0]
    output_json["json_ground_truth_index"] = j
    output_json["json_ground_truth"] = ground_truth_item
    output_json["json_extracted"] = {}
    # If not in it, it means llm didn't find a good match, so leave it empty.
    if "json_extracted_index" in output_json:
      if (
          str(output_json["json_extracted_index"]).isdigit()
          and int(output_json["json_extracted_index"]) > 0
          and int(output_json["json_extracted_index"])
          < len(json_model_response)
      ):
        output_json["json_extracted"] = json_model_response[
            int(output_json["json_extracted_index"])
        ]
      else:
        del output_json["json_extracted_index"]
    eval_list.append(output_json)
  return {
      "record_id": record_id,
      "ground_truth_length": len(json_ground_truth),
      "model_response_length": len(json_model_response),
      "response_json": eval_list,
  }

# %%


# %% [markdown]
# ### LLMSim dft

# %%
def get_lmsim_score_dft(prediction, reference,paper_id):
  return dft_domain_expert_model_based_eval(reference, prediction) # Client=None for external api


_METADATA_EVAL_PROMPT_FILENAME = file_root_path + "/prompts/dft_metadata_eval_output_1_shot.txt"
_STRUCTURE_EVAL_PROMPT_FILENAME = file_root_path + "/prompts/dft_structure_eval_output_1_shot.txt"


def get_dft_model_response_field(
    model_output_value: dict[str, Any], field_name: str
) -> list[Any] | str:
  """Returns the model response for a given field from the inference output.

     This applies to json responses from the dft chained inference output.

  Args:
    model_output_value: The model output response (from one two) as a dict.
    field_name: The name of the field to extract. This should be one of
      "structure_metadata", "dft_metadata", or "code".
  """
  if field_name in ["structure_metadata"]:
    if field_name in model_output_value:
      print(model_output_value)
      return model_output_value["structure_metadata"]
    else:
      return ""

  elif field_name in ["dft_metadata"]:
    if field_name in model_output_value:
      return model_output_value["dft_metadata"]
    else:
      return ""

  elif field_name == "code":
    if field_name in model_output_value:
      code = "\n".join(
          [x["code_element"] for x in model_output_value["code_elements"]]
      )
      code += "\n" + model_output_value["execution_code"]
      return code
    else:
      return ""

  raise ValueError(f"Unknown field name: {field_name}")


def get_annotated_structure_metadata_and_dft_params(
    gt_paper_code: str, verbose: int = 0
) -> dict[str, list[str]]:
  """Returns structure metadata and dft params from the ground truth code.

  Args:
    gt_paper_code: The ground truth code from .py file as a string.
    verbose: The verbosity level.
  """
  structures = []
  dft_params = []

  paper = gt_paper_code
  if "structure_metadata_" in paper:
    parts = paper.split("structure_metadata_")[1:]
    whole_parts = ["structure_metadata_" + part for part in parts]

    for part in whole_parts:
      if verbose > 1:
        print("PART:\n", part)
      if "parse_raw(" not in part:
        continue
      left, right, *_ = part.split("parse_raw(")
      if "StructureMetadata" in left:
        end_struc = ")"
        if "')" in right:
          end_struc = "')"
        elif "'\n)" in right:
          end_struc = "'\n)"
        struc_json = right.split(end_struc)[0].strip()
        if verbose > 0:
          print("Extracted structure:\n", struc_json)
        # clean_json = struc_json.replace('NaN', '"NaN"')
        # struc_json = ast.literal_eval(clean_json)
        structures.append(struc_json)

  if "dft_params_" in paper:
    parts = paper.split("dft_params_")[1:]
    # print(parts)
    whole_parts = ["dft_params_" + part for part in parts]

    for part in whole_parts:
      if verbose > 1:
        print("PART:\n", part)
      if "parse_raw(" not in part:
        continue
      left, right, *_ = part.split("parse_raw(")
      if "DFTParameters" in left:
        end_struc = ")"
        if "')" in right:
          end_struc = "')"
        elif "'\n)" in right:
          end_struc = "'\n)"
        dft_params_str = right.split(end_struc)[0].strip()
        if verbose > 0:
          print("Extracted dft_param:\n", dft_params_str)
        # clean_json = dft_params_str.replace('NaN', '"NaN"')
        # dft_params_str = ast.literal_eval(clean_json)
        dft_params.append(dft_params_str)

  gt_struc_jsons = []
  for struct_metadata in structures:
    gt_json = struct_metadata.split("'")[1]
    clean_json = gt_json.replace("NaN", '"NaN"')
    try:
      gt_structure_json = ast.literal_eval(clean_json)
      gt_struc_jsons.append(gt_structure_json)
    except Exception:  # pylint: disable=broad-exception-caught
      gt_struc_jsons.append(clean_json)

  gt_dft_params_jsons = []
  for dft_param in dft_params:
    gt_json = dft_param.split("'")[1]
    clean_json = gt_json.replace("NaN", '"NaN"')
    try:
      gt_dft_json = ast.literal_eval(clean_json)
      gt_dft_params_jsons.append(gt_dft_json)
    except Exception:  # pylint: disable=broad-exception-caught
      gt_dft_params_jsons.append(clean_json)

  return {
      "structures_metadata": gt_struc_jsons,
      "dft_params": gt_dft_params_jsons,
  }


def get_material_composition_from_struc(
    structure: str | dict[str, str],
) -> str | None:
  """Returns material composition from the structure metadata dict or string."""
  if isinstance(structure, dict):
    if "composition" in structure:
      return structure["composition"]
  elif isinstance(structure, str):
    if "composition" in structure:
      material = structure.split(r"\"composition\":")[1].split(",")[0]
      return material
  return None


def get_json_from_str(input_str: str) -> dict[str, Any] | None:
  """Returns the json object from the ground truth input string."""
  output_val = input_str.replace("NaN", '"NaN"')
  output_val = output_val.replace("true", '"1.0"')
  output_val = output_val.replace("false", '"NaN"')
  try:
    output_val = ast.literal_eval(output_val)
  except ValueError:
    return None
  return output_val


def parse_ground_truth_dft(ground_truth: str, client: Any) -> list[dict[str, Any]]:
  """Parses ground truth."""
  try:
    json_ground_truth = json5.loads(
        ground_truth.replace("\n", "").replace("\\", "")
    )
    if json_ground_truth and isinstance(json_ground_truth[0], str):
      json_ground_truth = [json5.loads(item) for item in json_ground_truth]
  except Exception:  # pylint: disable=broad-except
    # print("***using llm to parse")
    _ , ground_truth = call_openai(
        messages=[
            {
                "role": "user",
                "content": "Extract ground truth json list from the following text.\n"
                + ground_truth
                + "\nMake sure to remove all backslashes for escape characters. Output"
                " the json list ONLY, without any explanation, prefix or suffix:\n",
            }
        ],
        client=openai_client,
       
    ) 
    # print("***llm_ground_truth:\n", ground_truth)
    json_ground_truth = json5.loads(
        ground_truth.replace("\n", "").replace('\\"', "").replace("\\", "")
    )

  return json_ground_truth


def parse_model_response_dft(
    model_response: str, client: Any, use_llm=False
) -> list[dict[str, Any]]:
  """Parses model response."""
  def remove_prefix_suffix(text):
    return text.replace("\n", "").removeprefix("```json").removesuffix("```json").removeprefix("```").removesuffix("```").removeprefix("`").removesuffix("`")
  if use_llm:
    _,model_response = call_openai(
          messages=[
              {
                  "role": "user",
                  "content": "Extract json list from the following text.\n"
                  + model_response
                  + "\nMake sure to remove all backslashes for escape characters. Output"
                  " the json list ONLY, without any explanation, prefix or suffix:\n",
              }
          ],
          client=openai_client,
          )
    
  response_text = remove_prefix_suffix(model_response)
  # response_text = model_response
  # try:
  #   try:
  #     formatted_text = re.sub(r"(?<=\w)'(?=\w|\s)", "\\'", response_text)
  #     if not formatted_text:
  #       formatted_text = "{}"
  #       print("Response_text is empty")
  #     json_model_response = json5.load(formatted_text)
  #   except Exception as e:  # pylint: disable=broad-except
  #     print("Skipping incomplete last item: ", e)
  #     # print("***", response_text)
  #     ind = [m.start() for m in re.finditer(r",\s*\{", response_text)][-1]
  #     json_model_response = json5.loads(response_text[:ind] + "]")
  # except:
  #     return parse_model_response_dft(model_response, client, use_llm=True) if not use_llm else []
  # return json_model_response

  try:
    formatted_text = re.sub(r"(?<=\w)'(?=\w|\s)", "\\'", response_text)
    if not formatted_text:
      formatted_text = "{}"
      print("Response_text is empty")
    json_model_response = json5.loads(formatted_text)
  except Exception as e:  # pylint: disable=broad-except
    # print("using llm to parse model output because: ", e)
    return parse_model_response_dft(model_response, client, use_llm=True) if not use_llm else []
  return json_model_response

def dft_model_eval_paper(
    record_id: str | None,
    ground_truth: str,
    model_response: str,
    eval_prompt: str,
    client: Any,
) -> dict[str, Any]:
  """Runs model evaluation on material properties for a single paper.

  Args:
    record_id: record id or paper id.
    ground_truth: ground truth list in str type.
    model_response: model response list in str type.
    eval_prompt: eval prompt.
    client: llm client.

  Returns:
    model eval response json.
  """
  json_ground_truth = parse_ground_truth_dft(ground_truth, client)
  json_model_response = parse_model_response_dft(model_response, client)
  return model_eval_json(
      record_id=record_id,
      json_ground_truth=json_ground_truth,
      json_model_response=json_model_response,
      eval_prompt=eval_prompt,
      client=client,
  )


def dft_metadata_domain_expert_model_based_eval(
    ground_truth: str,
    model_response: str,
    client: Any | None = None,
    eval_prompt: str = _METADATA_EVAL_PROMPT_FILENAME,
    verbose: bool = True,
) -> dict[str, Any]:
  return dft_domain_expert_model_based_eval(
      ground_truth=ground_truth,
      model_response=model_response,
      client=client,
      eval_prompt=eval_prompt,
      verbose=verbose,
  )


def dft_structure_domain_expert_model_based_eval(
    ground_truth: str,
    model_response: str,
    client: Any | None = None,
    eval_prompt: str = _STRUCTURE_EVAL_PROMPT_FILENAME,
    verbose: bool = True,
) -> dict[str, Any]:
  return dft_domain_expert_model_based_eval(
      ground_truth=ground_truth,
      model_response=model_response,
      client=client,
      eval_prompt=eval_prompt,
      verbose=verbose,
  )


def dft_domain_expert_model_based_eval(
    ground_truth: str,
    model_response: str,
    client: Any | None = None,
    eval_prompt: str = _METADATA_EVAL_PROMPT_FILENAME,
    verbose: bool = True,
) -> dict[str, Any]:
  """Runs model based eval on dft.

  Args:
    ground_truth: ground truth list in str type.
    model_response: model response list in str type.
    client: llm client.
    eval_prompt: eval prompt.
    verbose: whether to print out eval results.

  Returns:
    eval result.
  """
  if verbose:
    print("Model eval started...")
  eval_output_item = dft_model_eval_paper(
      record_id=None,
      ground_truth=ground_truth,
      model_response=model_response,
      eval_prompt=eval_prompt,
      client=client,
  )
  if verbose:
    print("Model eval finished.")
  eval_result = eval_overall_result(
      eval_output_item, verbose=verbose
  )
  if verbose:
    print("Eval results:\n", eval_result)
  return eval_result

# %% [markdown]
# ### LLMSim mpve

# %%
def get_lmsim_score_mpve(prediction, reference,
                         paper_id: str = "",
                         ):
  # TODO: Convert to external model client
  return mpve_domain_expert_model_based_eval(reference, prediction,client=None,paper_id=paper_id) # Client=None for external api


# TODO: Convert to Drive path
_EVAL_PROMPT_FILENAME = file_root_path + "/prompts/mat_eval_output_1_shot.txt"


def parse_ground_truth_mpve(ground_truth: str) -> list[dict[str, Any]]:
  json_ground_truth = json_repair.repair_json(ground_truth.replace("\n", ""),return_objects=True)
  json_ground_truth.sort(key=lambda x: x["index"])
  # remove unnecessary fields for model eval to prevent hallucination
  for item in json_ground_truth:
    if "index" in item:
      del item["index"]
    if "paper_id" in item:
      del item["paper_id"]
    if "synonyms" in item:
      del item["synonyms"]
  return json_ground_truth


def parse_model_response_mpve(model_response: str) -> list[dict[str, Any]]:
  """Parses model response.

  Args:
    model_response:

  Returns:
  """
  response_text = (
      model_response.replace("\n", "")
      .removeprefix(" ")
      .removesuffix(" ")
      .removeprefix("```json")
      .removesuffix("```json")
      .removeprefix("```")
      .removesuffix("```")
      .removeprefix("`")
      .removesuffix("`")
  )
  try:
    formatted_text = re.sub(r"(?<=\w)'(?=\w|\s)", "\\'", response_text)
    if not formatted_text:
      formatted_text = "{}"
      print("Response_text is empty")
    json_model_response = json5.load(formatted_text)
  except Exception as e:  # pylint: disable=broad-except
    # print("Skipping incomplete last item: ", e)
    # ind = [m.start() for m in re.finditer(r",\s*\{", response_text)][-1]
    # json_model_response = json_repair.repair_json(response_text,return_objects=True)
    # print("using llm to parse")
    _ , response_text = call_openai(
        messages=[
            {
                "role": "user",
                "content": "Extract model_response json list from the following text.\n"
                + response_text
                + "\nMake sure to remove all backslashes for escape characters. Output"
                " the json list ONLY, without any explanation, prefix or suffix:\n",
            }
        ],
        client=openai_client,
       
    )
    # print("llm_model_response:\n", response_text)
    json_model_response = json_repair.repair_json(
        response_text.replace("\n", "").replace('\\"', "").replace("\\", ""),
        return_objects=True,
    )
    if isinstance(json_model_response, dict):
      json_model_response = [json_model_response]
  return json_model_response
  
  if not response_text.strip():
    print("Warning (parse_model_response_mpve): Model response text is empty after stripping. Returning empty list.")
    return []

  parsed_object: Any = None
  try:
    # Attempt to repair and parse the JSON.
    formatted_text = re.sub(r"(?<=\w)'(?=\w|\s)", "\\'", response_text) 
    if not formatted_text.strip():
        print("Warning (parse_model_response_mpve): Model response text became empty after formatting. Returning empty list.")
        return []
    parsed_object = json_repair.repair_json(formatted_text, return_objects=True)
  except Exception as e_repair:
    print(f"Warning (parse_model_response_mpve): json_repair.repair_json failed: {e_repair}. Trying fallback with json5.loads.")
    try:
      # Fallback: try to load with json5 directly
      parsed_object = json5.loads(response_text)
    except Exception as e_json5:
      print(f"Warning (parse_model_response_mpve): Fallback json5.loads also failed: {e_json5}. Returning empty list.")
      return []
  
  # Ensure the output is list[dict[str, Any]]
  if isinstance(parsed_object, list):
    # Filter the list to ensure all elements are dictionaries
    # result_list = [item for item in parsed_object if isinstance(item, dict)]
    # if len(result_list) != len(parsed_object):
    #   malformed_items = [item for item in parsed_object if not isinstance(item, dict)]
    #   print(f"Warning (parse_model_response_mpve): Filtered out {len(malformed_items)} non-dictionary item(s) from model response list. Items: {malformed_items}")
    return parsed_object
  elif isinstance(parsed_object, dict):
    # If a single dictionary is returned, wrap it in a list
    print(f"Warning (parse_model_response_mpve): Parsed model response is a single dictionary. Wrapping it in a list.")
    return [parsed_object]
  else:
    # If parsed_object is neither a list nor a dict (e.g., int, str, None)
    print(f"Warning (parse_model_response_mpve): Parsed model response is of unexpected type {type(parsed_object)} (value: {str(parsed_object)[:100]}...). Returning empty list.")
    return []


def mpv_model_eval_paper(
    record_id: str | None,
    ground_truth: str,
    model_response: str,
    eval_prompt: str,
    client: Any,
) -> dict[str, Any]:
  """Runs model evaluation on material properties for a single paper.

  Args:
    record_id: record id or paper id.
    ground_truth: ground truth list in str type.
    model_response: model response list in str type.
    eval_prompt: eval prompt.
    client: llm client.

  Returns:
    model eval response json.
  """
  json_ground_truth = parse_ground_truth_mpve(ground_truth)
  json_model_response = parse_model_response_mpve(model_response)
  return model_eval_json(
      record_id=record_id,
      json_ground_truth=json_ground_truth,
      json_model_response=json_model_response,
      eval_prompt=eval_prompt,
      client=client,
  )


def filter_ground_truth_properties(ground_truth: str) -> list[dict[str, Any]]:
  """Filters ground truth to only keep the properties we want to evaluate.

  Args:
    ground_truth: ground truth list in str type.

  Returns:
    filtered ground truth list.
  """
  json_ground_truth = json5.loads(ground_truth)
  filtered_ground_truth = []
  valid_property_names = [
      "bandgap",
      "band gap",
      "gap energy",
      "energy gap",
      "refractive_index",
      "refractive index",
      "index of refraction",
      "n-value",
      "n value",
  ]
  for item in json_ground_truth:
    for valid_property_name in valid_property_names:
      if valid_property_name in item["property_name"].lower():
        filtered_ground_truth.append(item)
        break
  return filtered_ground_truth


def mpve_domain_expert_model_based_eval(
    ground_truth: str,
    model_response: str,
    client: Any | None = None,
    eval_prompt: str = _EVAL_PROMPT_FILENAME,
    verbose: bool = True,
    paper_id: str = "",
) -> dict[str, Any]:
  """Runs model based eval on material properties.

  Args:
    ground_truth: ground truth list in str type.
    model_response: model response list in str type.
    client: llm client.
    eval_prompt: eval prompt.
    verbose: whether to print out eval results.

  Returns:
    eval result.
  """
  if verbose:
    print(f"Model eval started... with paper_id: {paper_id}")
  eval_output_item = mpv_model_eval_paper(
      record_id=paper_id,
      ground_truth=ground_truth,
      model_response=model_response,
      eval_prompt=eval_prompt,
      client=client,
  )
  if verbose:
    import time
    # 输出当前北京时间
    beijing_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Model eval finished.in {beijing_time_str}")
  eval_result = eval_overall_result(
      eval_output_item, verbose=verbose
  )
  if verbose:
    print("Eval results:\n", eval_result)
  return eval_result

def load_matsci_prompt(filepath: str) -> str:
  """Loads matsci prompt.

  Args:
    filepath: filepath of prompt.

  Returns:
    Loaded prompt.
  """
  # return resources.GetResource(filepath).decode("utf-8").strip()
  with open(filepath, 'r') as file:
    text_content = file.read()
  return text_content.strip()







# %%
from logging import raiseExceptions
def get_annotated_structure_metadata_and_dft_params(
    gt_paper_code: str, verbose: int = 0
) -> dict[str, list[str]]:
  """Returns structure metadata and dft params from the ground truth code.

  Args:
    gt_paper_code: The ground truth code from .py file as a string.
    verbose: The verbosity level.
  """
  structures = []
  dft_params = []

  paper = gt_paper_code
  if "structure_metadata_" in paper:
    parts = paper.split("structure_metadata_")[1:]
    whole_parts = ["structure_metadata_" + part for part in parts]

    for part in whole_parts:
      if verbose > 1:
        print("PART:\n", part)
      if "parse_raw(" not in part:
        continue
      left, right, *_ = part.split("parse_raw(")
      if "StructureMetadata" in left:
        end_struc = ")"
        if "')" in right:
          end_struc = "')"
        elif "'\n)" in right:
          end_struc = "'\n)"
        struc_json = right.split(end_struc)[0].strip()
        if verbose > 0:
          print("Extracted structure:\n", struc_json)
        # clean_json = struc_json.replace('NaN', '"NaN"')
        # struc_json = ast.literal_eval(clean_json)
        structures.append(struc_json)

  if "dft_params_" in paper:
    parts = paper.split("dft_params_")[1:]
    # print(parts)
    whole_parts = ["dft_params_" + part for part in parts]

    for part in whole_parts:
      if verbose > 1:
        print("PART:\n", part)
      if "parse_raw(" not in part:
        continue
      left, right, *_ = part.split("parse_raw(")
      if "DFTParameters" in left:
        end_struc = ")"
        if "')" in right:
          end_struc = "')"
        elif "'\n)" in right:
          end_struc = "'\n)"
        dft_params_str = right.split(end_struc)[0].strip()
        if verbose > 0:
          print("Extracted dft_param:\n", dft_params_str)
        # clean_json = dft_params_str.replace('NaN', '"NaN"')
        # dft_params_str = ast.literal_eval(clean_json)
        dft_params.append(dft_params_str)

  gt_struc_jsons = []
  for struct_metadata in structures:
    gt_json = struct_metadata.split("'")[1]
    clean_json = gt_json.replace("NaN", '"NaN"')
    try:
      gt_structure_json = ast.literal_eval(clean_json)
      gt_struc_jsons.append(gt_structure_json)
    except Exception:  # pylint: disable=broad-exception-caught
      gt_struc_jsons.append(clean_json)

  gt_dft_params_jsons = []
  for dft_param in dft_params:
    gt_json = dft_param.split("'")[1]
    clean_json = gt_json.replace("NaN", '"NaN"')
    try:
      gt_dft_json = ast.literal_eval(clean_json)
      gt_dft_params_jsons.append(gt_dft_json)
    except Exception:  # pylint: disable=broad-exception-caught
      gt_dft_params_jsons.append(clean_json)

  return {
      "structures_metadata": gt_struc_jsons,
      "dft_params": gt_dft_params_jsons,
  }


def preprocess_ground_truth(
    ground_truth: str, task_name: str, prompt: str
) -> str:
  """Preprocesses the ground truth before sending to eval."""
  # Drops the record_ids.
  json_gt = json5.loads(ground_truth)
  if isinstance(json_gt, dict):
    json_gt.pop("record_id", None)
    json_gt.pop("arxiv_id", None)
    json_gt.pop("paper_id", None)
  if isinstance(json_gt, list):
    for item in json_gt:
      if isinstance(item, dict):
        item.pop("record_id", None)
        item.pop("arxiv_id", None)
        item.pop("paper_id", None)
  groundtruth_with_no_ids = json5.dumps(json_gt)
  # Preprocess for dft metadata tasks.
  if task_name == "dft" and prompt == "extract_dft_metadata_1_shot":
    processed = get_annotated_structure_metadata_and_dft_params(
        groundtruth_with_no_ids
    )["dft_params"]
  elif task_name == "dft" and prompt == "extract_structure_data_1_shot":
    processed = get_annotated_structure_metadata_and_dft_params(
        groundtruth_with_no_ids
    )["structures_metadata"]
  else:
    processed = groundtruth_with_no_ids
  return str(processed)


def read_task_ground_truth_and_response(
    ground_truth_path: str,
    model_response_path: str,
) -> Tuple[str, str, str]:
  """Reads in the ground truth and response for all tasks."""
  try:
    # Gets the ground truth.
    with open(ground_truth_path, "r") as f:
      ground_truth_info = f.read()

    model_response = ""
    inf_prompt = ""
    if os.path.exists(model_response_path):
      with open(model_response_path, "r") as f:
        full_model_response = json5.loads(f.read())
        if "response_text" in full_model_response:
          model_response = full_model_response["response_text"]
        else:
          raise ValueError(
              f"ERROR: The succeeded response for {model_response_path} does not contain response_text."
          )
        if 'pdb' in ground_truth_path:
          if 'prompt_text' in full_model_response:
            inf_prompt = full_model_response["prompt_text"]
          else:
            raise ValueError(
                f"ERROR: The succeeded response for {model_response_path} does not contain prompt_text."
            )

    failed_model_response = model_response_path.replace("success", "failure")
    # Gets the response.
    exception_message = ""
    if os.path.exists(failed_model_response):
      with open(failed_model_response, "r") as f:
        full_model_response = json5.loads(f.read())
        if "exception_message" in full_model_response:
          exception_message = full_model_response["exception_message"]
        elif "command-r-plus" in failed_model_response and "response_text" in full_model_response:
          exception_message = full_model_response["response_text"]
        else:
          raise ValueError(
              f"ERROR: The failure response for {failed_model_response} does not contain exception message."
          )

    return ground_truth_info, model_response, exception_message, inf_prompt
  except Exception as e:
    print(f"ERROR: {e}")
    raise Exception(e)

# %% [markdown]
# ## static configs

# %%
# _SHARED_METRCS = [get_rouge_score, get_bert_score]
# _FULL_ADDITIONAL_METRICS = {
#     "pdb": {
#         "reconstruct_protein_amino_acid_sequence_0_shot": {
#             pdb_reconstruction_eval
#         },
#     },
#     "mpve": {
#         "mat_paper_to_property_1_shot": {
#             get_lmsim_score_mpve
#         },
#         "mat_paper_to_property_1_shot_exclude_trivia": {
#             get_lmsim_score_mpve
#         },
#         "mat_paper_to_property_1_shot_bandgap_refractive": {
#             get_lmsim_score_mpve
#         }
#     },
#     "dft": {
#         "extract_structure_data_1_shot": {
#             get_lmsim_score_dft
#         },
#         "extract_dft_metadata_1_shot": {
#             get_lmsim_score_dft
#         },
#     },
#     "biogr": {
#         "georeference_image_0_shot": {
#             biodiversity_georeferencing_eval
#         }
#     }
# }

# _PRIMARY_ADDITIONAL_METRICS = {
#     "pdb": {
#         "reconstruct_protein_amino_acid_sequence_0_shot": {
#             pdb_reconstruction_eval
#         },
#     },
#     "biogr": {
#         "georeference_image_0_shot": {
#             biodiversity_georeferencing_eval
#         }
#     }

# }

# _LLM_LIST = ["command-r-plus", "longllama", "mixtral-gcp",
#              "gemini-1.5-flash-latest", "gemini-1.0-pro", "gemini-1.5-pro-latest",
#              "gpt-4o", "claude-3-opus-20240229", 'gemini-2.0-flash-latest',
#              "deepseek-r1","doubao-1.5-thinking-pro"
#              ]

# _BIOGR_EXCLUDE_LLM = ["command-r-plus", "longllama", "mixtral-gcp"]
# _TASK_EVAL_CONFIGS = {
#     "hfe": {
#         "extract_hamiltonian_0_shot": {
#         }
#     },
#     "hfd": {
#         "derivation_prompt": {
#         }
#     },
#     "qecc_65": {
#         "describe_code_in_paper": {
#         }
#     },
#     "pdb": {
#         "reconstruct_protein_amino_acid_sequence_0_shot": {
#         }
#     },
#     "mpve": {
#         "mat_paper_to_property_1_shot": {
#         },
#         "mat_paper_to_property_1_shot_bandgap_refractive": {
#         },
#         "mat_paper_to_property_1_shot_exclude_trivia": {
#         },
#     },
#     "dft": {
#         "write_code_for_paper_0_shot": {
#         },
#         "extract_structure_data_1_shot": {
#         },
#         "extract_dft_metadata_1_shot": {
#         },
#     },
#     "geo": {
#         "extract_dataset_from_geo_papers_0_shot": {
#         }
#     },
#     "biogr": {
#         "georeference_image_0_shot": {
#         }
#     },
# }
# all_ids_per_task = {'pdb': ['1A12', '1A33', '1AIL', '1AOA', '1AQA', '1AZS', '1BL8', '1BM8', '1BXO', '1CC5', '1CDK', '1CGF', '1CLL', '1CTF', '1DIN', '1DXX', '1E9H', '1EST', '1F9V', '1G6X', '1GOF', '1GP1', '1HCG', '1HHO', '1HNF', '1HNV', '1IAV', '1IGM', '1IGT', '1JM1', '1KXQ', '1M03', '1M17', '1M8Q', '1MBG', '1MBO', '1MHC', '1NKO', '1POH', '1PRC', '1R09', '1RCP', '1RGS', '1SBT', '1SU4', '1TIT', '1TNK', '1UBQ', '2A99', '2ACE', '2AYN', '2J1N', '2POR', '2R6G', '3ADN', '3C7E', '3LCK', '3R2E', '4CPA', '5CPA', '5HZN', '7B3N', '7L1E', '7V8O'],
#                     'biogr': ['10212153_1', '260729_2', '531730_1', '556058_2', '563682_1', '564487_1', '575983_2', '578304_2', '583537_1', '585031_1', '587419_1', '590257_1', '591038_1', '592166_1', '592526_1', '592805_4', '594665_1', 'S0048969724009641_1', 'S1470160X22006951_1', 'a_decade_of_submersible_observations_1', 'a_new_species_of_river_1', 'a_preliminary_investigation_of_the_3', 'a_simple_genetic_method_to_1', 'a_small_warm_tributary_provides_1', 'abundance_of_longbilled_curlews_on_1a', 'an_overview_of_marine_biodiversity_3', 'an_overview_of_marine_biodiversity_8', 'assessment_of_ambystomatid_salamander_populations_1', 'assessment_of_potential_recovery_viability_2', 'availability_of_supplemental_corn_for_1', 'barriers_to_gene_flow_in_1', 'baseline_assessments_for_coral_reef_1', 'bat_predation_by_spiders_1', 'biotic_assemblages_of_gelatinous_zooplankton_3', 'bird_monitoring_at_effigy_mounds_1', 'birds_of_the_kilbuck_and_1', 'breeding_population_size_of_the_1', 'ceratonova_shasta_infection_in_lower_1', 'characterization_of_a_developing_recreational_1', 'chewing_lice_of_swan_geese_1', 'comparison_of_endoparasite_abundance_and_1', 'comparisons_of_walleye_fecundity_before_1', 'conservation_genetics_of_the_endangered_1', 'cooccurrence_of_ecologically_similar_species_1', 'deep_vs_shallow_gps_tags_1', 'density_of_axis_deer_in_1', 'despite_regional_variation_gymnorhinus_cyanocephalus_1', 'distribution_abundance_and_breeding_activities_1', 'distribution_and_abundance_of_least_3', 'distribution_morphology_and_karyotype_of_1', 'diurnal_human_activity_and_introduced_1', 'diving_patterns_and_foraging_locations_1', 'dna_barcoding_the_native_flowering_5', 'documentation_of_a_probable_spawning_1', 'ece310733_1', 'energy_density_of_three_prosopium_1', 'environment_affects_sucker_catch_rate_1', 'evaluating_spatial_coverage_of_the_1', 'evening_bats_captured_in_a_1', 'expansion_of_smallmouth_bass_distribution_1', 'extreme_wildlife_declines_and_concurrent_1', 'fecal_genotyping_to_estimate_small_1', 'first_record_of_paronatrema_vaginicola_1', 'fish_predation_by_semiaquatic_spiders_1', 'foraging_ecology_of_southern_sea_1', 'four_centuries_of_change_in_1', 'global_conservation_priorities_for_marine_4', 'habitat_suitability_assessment_for_tule_4', 'habitat_use_and_reproductive_success_1', 'hawaiian_hoary_bat_lasiurus_cinereus_1', 'hiding_in_plain_sight_federally_1', 'high_similarity_in_winter_diet_1', 'impacts_of_the_czu_lightning_1', 'incidental_take_of_giant_sea_7', 'incorporating_expanded_sampling_into_an_1', 'inventory_of_eelgrass_zostera_marina_1', 'jwmg22383_1', 'larval_and_juvenile_longfin_smelt_1', 'leveraging_angler_effort_to_inform_1', 'longterm_occupancy_monitoring_reveals_value_1', 'machine_learning_to_understand_patterns_1', 'macrohabitat_suitability_model_for_the_1', 'macroscale_effects_of_the_monument_1', 'marine_biodiversity_in_the_atlantic_4', 'microhabitat_characteristics_and_management_of_1', 'monitoring_fiveneedle_pine_on_bureau_1', 'monitoring_nesting_waterbirds_for_the_1', 'monitoring_questing_winter_tick_abundance_1', 'movement_patterns_of_two_bat_1', 'natal_contributions_of_kokanee_salmon_1', 'occurrence_of_a_reproducing_wild_3', 'occurrence_of_batrachochytrium_dendrobatidis_in_1', 'onceiconic_pismo_clams_persist_in_1', 'patterns_of_florida_bonneted_bat_2', 'population_and_spatial_dynamics_of_1', 'population_density_and_habitat_selection_2', 'population_genomic_surveys_for_six_1', 'postfire_survival_of_the_threatened_4', 'rangewide_genetic_analysis_of_an_1', 'rapid_population_decline_in_mckays_1', 'recovering_the_lost_potential_of_5', 'relative_influence_of_environmental_factors_1', 'rescuing_and_monitoring_white_sturgeon_2', 'revealing_biases_in_insect_observations_5', 'review_of_considerations_for_restoration_1', 'road_and_highway_undercrossings_as_1', 'roseate_tern_breeding_dispersal_and_1', 's41598022209644_1', 's41598023276709_1', 's4159802334533_1', 'sampling_duration_and_season_recommendations_1', 'sea_level_rise_vulnerability_assessment_1', 'seacliff_bedstraw_galium_buxifolium_patterns_4', 'seasonal_and_spatial_distribution_of_1', 'spatial_relationships_and_mesoscale_habitat_1', 'spatial_variation_in_density_of_1', 'status_and_distribution_of_arroyo_2', 'status_assessment_of_the_endangered_1', 'status_of_landbirds_in_the_1', 'striped_bass_movement_in_a_1', 'syntopy_in_california_redlegged_and_1', 'testing_a_singlevisit_sampling_approach_1', 'the_biodiversity_of_the_mediterranean_5', 'the_first_occurrence_of_the_3', 'the_importance_of_subarctic_intertidal_1', 'the_lion_in_west_africa_1', 'time_series_modeling_of_rainfall_1', 'trace_elements_in_blood_of_1', 'travel_management_planning_for_wildlife_1', 'trends_in_amphibian_occupancy_in_1', 'tricolored_blackbird_survey_methods_1', 'tule_elk_selection_of_surface_1', 'unintended_consequences_of_species_translocations_1', 'us_atlantic_and_gulf_of_1', 'use_of_aerial_distance_sampling_1', 'utilizing_the_timetoevent_framework_to_1', 'validating_a_nonlethal_method_of_1', 'western_purple_martin_progne_subis_1'],
#                     'geo': ['00000', '14614a88b3e44e601c5cf8f71b5e07ca989beb0b', '213d2232a49507f81b4e17e50de7675c88fbc672', '33b0925f7681f3199a5d075324e7f3c5e33f2c76', '41f20bb04729a55ca9c2eaf579adf3ed5729044b', '54a9885771350f38135f30f43ef874e0a30be07b', '5c37c2aa2e108e17e37c6db29a4e5afe6a811119', '7dc47696eb876d85a3dfc6884f61fa8832d5e5e8', '83a1a10e3a2416e1d93bc3dbb482db4ccb707eda', '850ca33e8c1853c1735da63073ec3910bce91ddc', '9bdabc37e4af91c4fb53e205502204b510e3b972', 'a09b49e5f2c6b818e479bd29343eae9005f8ca26', 'ab6d648944f306fa1e2d275115b94d36478d9d2a', 'b90358f971e19a60c305acff2867b89dd197fdf6', 'c3a3a5a24206a9b38d9f4727f78cc8f323e398b2', 'e57ae1987bce88add50696843c8979456ce55561', 'e88aa0bccedc5f07bfee8f2db7a85351e65ec24a', 'e900993457d4d256cbfbe8a7527b6745f130a98e', 'e9c8932d5fcdf067821f8bf24b7462e5c7f73054'], 'hfd': ['1010.1819', '1106.6060', '1208.0116', '1310.2674', '1508.00296', '1812.04213', '2004.04168', '2008.08998', '2012.04554', '2108.02159', '2110.11330', '2111.01152', '2112.07523', '2308.03843', '2308.07488'], 'hfe': ['1010.1819', '1112.4222', '1202.4956', '1206.0608', '1208.0116', '1212.5363', '1401.2167', '1506.01488', '1507.06420', '1510.06887', '1512.02398', '1601.00996', '1812.04213', '1908.05417', '2007.15166', '2008.08998', '2102.13507', '2108.02159', '2111.09813', '2112.07523', '2206.10024', '2208.07620', '2209.15374', '2210.06674', '2210.08025', '2210.14517', '2302.04864', '2303.09821', '2303.18025', '2306.02127', '2306.12486', '2307.03793', '2307.04307', '2307.07531', '2307.11810', '2308.01997', '2308.03843', '2311.13191'], 'qecc_65': ['1501.07779', '1502.05267', '1503.06237', '1503.08800', '1505.02576', '1602.00008', '1603.04442', '1604.07925', '1703.02973', '1707.02308', '1708.08474', '1709.04471', '1709.08658', '1710.04631', '1712.07666', '1801.05897', '1802.07419', '1805.01474', '1809.09801', '1903.03937', '1906.11394', '1907.09528', '1910.10746', '2003.02717', '2007.09154', '2007.12152', '2008.09495', '2009.03921', '2010.06628', '2106.02649', '2107.02194', '2110.11510', '2112.01446', '2201.07802', '2203.00103', '2203.16534', '2209.11405', '2210.10808', '2210.16957', '2212.09935', '2303.02432', '2303.04798', '2306.11621', '2309.16503', '2311.07679', '2311.08653', '2311.13040', '2312.04522', '2402.07476', 'cond-mat_0010440', 'cond-mat_0607736', 'cond-mat_9707273', 'cs_0509062', 'quant-ph_0008040', 'quant-ph_0210097', 'quant-ph_0502086', 'quant-ph_0605138', 'quant-ph_0701020', 'quant-ph_0702075', 'quant-ph_9703002', 'quant-ph_9705052', 'quant-ph_9711021', 'quant-ph_9711049', 'quant-ph_9810055', 'quant-ph_9906114'], 'dft': ['2023_09_22_01b9cdba467fd7882e42g', '2023_09_22_07b4d66e23971ccb85c0g', '2023_09_22_0ce1b5ea9a8637db5435g', '2023_09_22_109a6cd5d015ce89e7f3g', '2023_09_22_12cb692c6b82606615a6g', '2023_09_22_13bcf90c3ef43f1413deg', '2023_09_22_14ab0a44fc8fc33fa338g', '2023_09_22_14ddd903c0b77b5e50c2g', '2023_09_22_182e0132c513bc81a414g', '2023_09_22_1a3a4803f9d9cec16d38g', '2023_09_22_1af7e6342ebcf1ce3ea5g', '2023_09_22_1b00ed3a142a7a1b2582g', '2023_09_22_1def59bea80d6e9f67ccg', '2023_09_22_22c19cfc32d2575f9a52g', '2023_09_22_24d7d9ed97e042af9f29g', '2023_09_22_2d35eebe2e85cecf4103g', '2023_09_22_2d44dda253969d6ce7f6g', '2023_09_22_2e5b7c3f50b6a643e33ag', '2023_09_22_2e9b1b9bffe7fd47b18fg', '2023_09_22_39b7d4444fd6ff852b12g', '2023_09_22_3ae1fab1e33569c30b8dg', '2023_09_22_430c3ddffa99af6a2545g', '2023_09_22_433b6bb3bfc2391f7300g', '2023_09_22_48fb54662c601e035a91g', '2023_09_22_49d6cc9c5e7a469afee0g', '2023_09_22_4eeeb89f0a6f52e58610g', '2023_09_22_4ef39bb4116ac1dad9e9g', '2023_09_22_5237e17b3b341fecc9d9g', '2023_09_22_54050a76c33463a8157fg', '2023_09_22_55f975d508230ef05caeg', '2023_09_22_63a752c620bbc784200cg', '2023_09_22_67e874a3c664208e2d2fg', '2023_09_22_6ac063e0fb85cfc10dd0g', '2023_09_22_70326f83ce0dcec87b50g', '2023_09_22_7433a6c7542334063731g', '2023_09_22_77a765fcab6029c666b4g', '2023_09_22_799ee1c298d190145c70g', '2023_09_22_7c5bbe7e076779b790ccg', '2023_09_22_7c76b066edb1f6f53739g', '2023_09_22_7dab45b11a3ede362147g', '2023_09_22_8181c4d6fa78a2ea82dag', '2023_09_22_81b3dfeb3597db5200c5g', '2023_09_22_84af4abb781aeead403eg', '2023_09_22_87d405f182ae3706ea0cg', '2023_09_22_8b46d7b3e561e7f28495g', '2023_09_22_8df0b56e310badc55de3g', '2023_09_22_900c617369212d6bc72fg', '2023_09_22_910aca2a500d3bf9bf47g', '2023_09_22_980121c407cbdaa46afdg', '2023_09_22_984f5b905c02b6f21733g', '2023_09_22_995805c76f676dddab4fg', '2023_09_22_9a007c3865721f379b39g', '2023_09_22_9a591ebf98377fd0ebe2g', '2023_09_22_9e2bc88db643c6ba8aa0g', '2023_09_22_a0271d2dc7b0f2506498g', '2023_09_22_b0501f9057db320b8ad9g', '2023_09_22_b2865949a80ad08a2835g', '2023_09_22_bba019fc933fc84ad347g', '2023_09_22_cb81fee8faa69f4d7078g', '2023_09_22_cc792b66f9a5779f9798g', '2023_09_22_cff3389f103b8f7971d0g', '2023_09_22_d84d81c022c4b2981048g', '2023_09_22_d90e94cbd96e4b6ddb8bg', '2023_09_22_dd9f0f77c116dc99583ag', '2023_09_22_ddfb75e0fb765dc682bbg', '2023_09_22_e06d11a6e698afe5f2d7g', '2023_09_22_e32d0198a1f3dddb5ba2g', '2023_09_22_e69f3d7ce6c4ff487115g', '2023_09_22_e8d1bc2fb9f3dce5f341g', '2023_09_22_edae82c7fbe0c4062118g', '2023_09_22_efbe854b8da1545fbe9bg', '2023_09_22_f752dc9d5ac72657e3f5g', '2023_09_22_f7714e3a468c91c6f56ag', '2023_09_22_f8875eb68affb0a6cb2bg'], 'mpve': ['10222315', '11093908', '11181068', '12841719', '135893324', '137261967', '137362119', '15804005', '17645319', '2837337', '317542', '53384093', '53519111', '55005437', '6183251', '68518', '97574650']}

# %% [markdown]
# # get full eval results
# 
# 

# %% [markdown]
# ## calculates metrics

# %%
# Set this to True if you want all metrics including LLMSim scores.
runs_full_metric = False #@param

# %%
# results_json = {}
# for task in _TASK_EVAL_CONFIGS.keys():
#   results_json[task] = {}
#   for prompt in _TASK_EVAL_CONFIGS[task]:
#     results_json[task][prompt] = {}
#     for llm_name in _LLM_LIST:
#       results_json[task][prompt][llm_name] = {}
#       for record_id in all_ids_per_task[task]:
#         print(f"task: {task}, prompt: {prompt}, llm_name: {llm_name}, record_id: {record_id}")
#         results_json[task][prompt][llm_name][record_id] = {}
#         print(f"{task}, {prompt}, {llm_name}, {record_id}")
#         gt_path = os.path.join(file_root_path, "data", task, "ground_truth", record_id + ".json")
#         model_response_path = os.path.join(file_root_path, "inference_outputs", task, prompt, llm_name, "success", record_id + ".json")
#         try:
#           ground_truth_info, model_response, exception_message, inf_prompt = read_task_ground_truth_and_response(gt_path, model_response_path)
#           ground_truth_info = preprocess_ground_truth(ground_truth_info, task, prompt)
#         except Exception as e:
#           print(f"ERROR: {e}")
#           continue
#         if (not task == 'pdb' and model_response) or (task == 'pdb' and model_response and inf_prompt):

#           full_additional_metrics = list(_FULL_ADDITIONAL_METRICS[task][prompt]) if task in _FULL_ADDITIONAL_METRICS  and prompt in _FULL_ADDITIONAL_METRICS[task] else []
#           primary_additional_metrics = list(_PRIMARY_ADDITIONAL_METRICS[task][prompt]) if task in _PRIMARY_ADDITIONAL_METRICS  and prompt in _PRIMARY_ADDITIONAL_METRICS[task] else []
#           additional_metrics = full_additional_metrics if runs_full_metric else primary_additional_metrics
#           all_metrics = additional_metrics + _SHARED_METRCS
#           for metric in all_metrics:
#             try:
#               if task == 'pdb' and metric in additional_metrics:
#                 res = metric(model_response, ground_truth_info, inf_prompt)
#               else:
#                 res = metric(model_response, ground_truth_info)
#               print(res)
#               results_json[task][prompt][llm_name][record_id].update(res)
#             except Exception as e:
#               print("##### ERROR #####")
#               print(e)
#               print(f"##### skipped {task}, {prompt}, {llm_name}, {record_id} #####")
#               continue
import concurrent.futures
def run_evaluation_signal_model(TASK_NAME, INFER_MODEL_NAME):

  
  if TASK_NAME == "MPVE":
    inference_path=os.path.join(file_root_path, "inference","multi_runs/current/mpve/mat_paper_to_property_1_shot_exclude_trivia",INFER_MODEL_NAME,"run_1/success")
    # print(os.listdir(inference_path))
    # 获得去掉后缀的文件名
    paper_ids = [os.path.splitext(file)[0] for file in os.listdir(inference_path)]
    print(paper_ids)
    ground_truth_path = os.path.join(file_root_path, "data", "mpve", "ground_truth")
  elif TASK_NAME == "DFT":
    inference_path=os.path.join(file_root_path, "inference","multi_runs/current/dft/extract_dft_metadata_1_shot",INFER_MODEL_NAME,"run_1/success")
    # print(os.listdir(inference_path))
    # 获得去掉后缀的文件名
    paper_ids = [os.path.splitext(file)[0] for file in os.listdir(inference_path)]
    print(paper_ids)
    ground_truth_path = os.path.join(file_root_path, "data", "dft", "ground_truth")
  # paper_ids=['11093908']

  # print(os.listdir(ground_truth_path))
  # for paper_id in paper_ids:
  #   # print(paper_id)
  #   gt_path = os.path.join(ground_truth_path, paper_id + ".json")
  #   model_response_path = os.path.join(inference_path, paper_id + ".json")
  #   try:
  #     ground_truth_info, model_response, exception_message, inf_prompt = read_task_ground_truth_and_response(gt_path, model_response_path)
  #     ground_truth_info = preprocess_ground_truth(ground_truth_info, "mpve", "mat_paper_to_property_1_shot_exclude_trivia")
  #   except Exception as e:
  #     print(f"ERROR: {e}")
  #     continue
  #   if (not "mpve" == 'pdb' and model_response) or ("mpve" == 'pdb' and model_response and inf_prompt):
  #     res = get_lmsim_score_mpve(model_response, ground_truth_info)
  #     print(res)

  def process_paper_id(paper_id):
    gt_path = os.path.join(ground_truth_path, paper_id + ".json")
    model_response_path = os.path.join(inference_path, paper_id + ".json")
    try:
      ground_truth_info, model_response, exception_message, inf_prompt = read_task_ground_truth_and_response(gt_path, model_response_path)
      if TASK_NAME == "MPVE":
        ground_truth_info = preprocess_ground_truth(ground_truth_info, "mpve", "mat_paper_to_property_1_shot_exclude_trivia")
      elif TASK_NAME == "DFT":
        ground_truth_info = preprocess_ground_truth(ground_truth_info, "dft", "extract_dft_metadata_1_shot")  
      else:
        raise ValueError(f"ERROR: Unsupported task name {TASK_NAME}.")
    except Exception as e:
      print(f"ERROR: {e}")
      return None
    if (not "mpve" == 'pdb' and model_response) or ("mpve" == 'pdb' and model_response and inf_prompt):
      if TASK_NAME == "MPVE":
        res = get_lmsim_score_mpve(model_response, ground_truth_info,paper_id=paper_id)
      elif TASK_NAME == "DFT":
        res = get_lmsim_score_dft(model_response, ground_truth_info,paper_id=paper_id)
      return res
    return None
  #使用ThreadPoolExecutor来并行处理文件
  # paper_ids=["53519111"]
  # paper_ids = paper_ids[:1]
  with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_paper_id, paper_ids))
  # results = []
  # error_records = []
  # for paper_id in paper_ids:
  #   result = process_paper_id(paper_id)
  #   results.append(result)
  # 处理结果
  for paper_id, result in zip(paper_ids, results):
    if result is not None:
      print(f"Paper ID: {paper_id}, Result: {result}")
    else: 
      print(f"Paper ID: {paper_id} procNoneessing failed.")



  # 给rusults加上paper_id，整体保存为一个json文件，存到inference_path
  import json
  for result,paper_id in zip(results, paper_ids):
      if result is not None:
          result["paper_id"] = paper_id
          print(result)
          # result["eval_model"] = EVAL_MODEL_NAME
      else:
          print(f"Paper ID: {paper_id} processing failed.")
  # Save the results to a JSON file

  output_file = os.path.join(inference_path, "..",f"{INFER_MODEL_NAME} results.json")
  with open(output_file, "w") as f:
      json.dump(results, f, indent=4, ensure_ascii=False)


  f1=0.0
  precision=0
  recall=0
  valid_num=0
  for data in results:
      if data['num_ground_truth'] == 0:
          print(f"Paper ID: {data['paper_id']} has no ground truth.")
          continue
      f1+=data["f1"]
      precision+=data["precision"]
      recall+=data["recall"]
      valid_num+=1
  print(INFER_MODEL_NAME)
  print(TASK_NAME)
  print(f1/valid_num)
  print(precision/valid_num)
  print(recall/valid_num)
  # Save the results to a JSON file
  output_file = os.path.join(inference_path, "..", f"{INFER_MODEL_NAME} ave_accuracy.json")
  with open(output_file, "w") as f:
      json.dump({"f1":f1/valid_num,"precision":precision/valid_num,"recall":recall/valid_num}, f, indent=4, ensure_ascii=False)
      
def run_eval():
  for task_name in task_list:
    # task_name= "MPVE"
    for infer_model_name in model_list:
      # run_evaluation("MPVE", infer_model_name)
      run_evaluation_signal_model(task_name, infer_model_name)

# with concurrent.futures.ThreadPoolExecutor() as executor:
#   for infer_model_name in model_list:
#     # run_evaluation("MPVE", infer_model_name)
#     executor.submit(run_evaluation, "DFT", infer_model_name)
if __name__ == "__main__":
  run_eval()
