# %% [markdown]
# # imports

# %%
import json5
import re
import functools
from typing import Any, Tuple, Union, Dict
# from Bio import Align
import glob
# from bert_score import score
# from rouge_score import rouge_scorer
import numpy as np
import ast
import os
# import google.generativeai as genai
# import Levenshtein
# MODEL_NAME = "deepseek-r1"
# "doubao-1.5-thinking-pro"
EVAL_MODEL_NAME = "deepseek-r1"
INFER_MODEL_NAME = "doubao-1.5-thinking-pro"
file_root_path="."

# %% [markdown]
# # eval functions

# %% [markdown]
# ### huoshan

# %%
import time
import yaml
import requests
import traceback
def call_huoshan(messages, model_name="doubao-1.5-thinking-pro", config_path="../colabs/api_config.yaml"):
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

# %% [markdown]
# ### LLMSim util

# %%
# @functools.lru_cache(maxsize=1)
# def get_model(model_name: str = 'gemini-1.5-pro-latest'):

#   return genai.GenerativeModel(model_name=model_name)


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
      model_response_item["json_extracted_index"] = k
    prompt = (
        load_matsci_prompt(eval_prompt)
        .replace(
            "{{json_ground_truth}}", json5.dumps(ground_truth_item, indent=2)
        )
        .replace(
            "{{json_extracted_list}}", json5.dumps(json_model_response, indent=2)
        )
    )

    
    _ , output = call_huoshan(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model_name=EVAL_MODEL_NAME,
        config_path=os.path.join(file_root_path, "colabs", "api_config.yaml"),
    )
    try:
      output_json = json5.loads(output)
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


# %% [markdown]
# ### LLMSim mpve

# %%
def get_lmsim_score_mpve(prediction, reference,
                         paper_id: str = "",
                         ):
  # TODO: Convert to external model client
  return mpve_domain_expert_model_based_eval(reference, prediction, client=None,paper_id=paper_id) # Client=None for external api


# TODO: Convert to Drive path
_EVAL_PROMPT_FILENAME = file_root_path + "/prompts/mat_eval_output_1_shot.txt"


def parse_ground_truth_mpve(ground_truth: str) -> list[dict[str, Any]]:
  json_ground_truth = json5.loads(ground_truth.replace("\n", ""))
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
    json_model_response = json5.loads(formatted_text)
  except Exception as e:  # pylint: disable=broad-except
    print("Skipping incomplete last item: ", e)
    ind = [m.start() for m in re.finditer(r",\s*\{", response_text)][-1]
    json_model_response = json5.loads(response_text[:ind] + "]")
  return json_model_response


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
    print("Model eval started...")
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
  pre = min(num_match / num_response if num_response else np.nan, 1.0)
  rec = min(num_match / num_gt if num_gt else np.nan, 1.0)
  return {
      "num_match": num_match,
      "num_ground_truth": num_gt,
      "num_model_response": num_response,
      "precision": pre,
      "recall": rec,
      "f1": 2.0 * pre * rec / (pre + rec) if pre + rec else 0.0,
  }


# %%
def center_of_bbox(json_coords: dict[str, float]) -> dict[str, float]:
  # NOTE: This doesn't work if you wrap around the 180 --> -179
  # longitude line in the Pacific but we don't have it in our data.
  bbox_center = {}
  bbox_center["lat"] = np.mean([json_coords["S"], json_coords["N"]])
  bbox_center["lng"] = np.mean([json_coords["E"], json_coords["W"]])
  return bbox_center


def compute_distance(
    lat1_deg: float, lng1_deg: float, lat2_deg: float, lng2_deg: float
) -> float:
  """Computes the distance between two points on a sphere in meters.

  Args:
    lat1_deg: Latitude of the first point in degrees.
    lng1_deg: Longitude of the first point in degrees.
    lat2_deg: Latitude of the second point in degrees.
    lng2_deg: Longitude of the second point in degrees.

  Returns:
    The distance between the two points in meters.
  """
  # Haversine Formula for the geodesic distance on a sphere.
  # See https://en.wikipedia.org/wiki/Haversine_formula.
  lat1 = np.deg2rad(lat1_deg)
  lng1 = np.deg2rad(lng1_deg)
  lat2 = np.deg2rad(lat2_deg)
  lng2 = np.deg2rad(lng2_deg)

  alpha = np.sin((lat2 - lat1) * 0.5)
  gamma = np.sin((lng2 - lng1) * 0.5)
  alpha = alpha * alpha + np.cos(lat1) * np.cos(lat2) * gamma * gamma
  if alpha > 1.0:
    alpha = 1.0  # bulletproof sqrt(1-alpha)
  gamma = 2.0 * np.arctan2(np.sqrt(alpha), np.sqrt(1.0 - alpha))
  return 6371000 * gamma


def compute_center_error_km(
    center_prediction: dict[str, float], center_ground_truth: dict[str, float]
) -> float:
  return 0.001 * compute_distance(
      center_prediction["lat"],
      center_prediction["lng"],
      center_ground_truth["lat"],
      center_ground_truth["lng"],
  )


def compute_box_size_km(coords: dict[str, float]) -> float:
  # Returns half of the diagonal (e.g. like half of a TV). This corresponds
  # to the radius of a circle that inscribes the rectangle.
  return (
      0.001
      * 0.5
      * compute_distance(coords["S"], coords["W"], coords["N"], coords["E"])
  )


def compute_distance_metrics(
    prediction: dict[str, float], ground_truth: dict[str, float]
) -> dict[str, float]:
  """Computes distance metrics between the prediction and ground truth.

  Computes two distance metrics:
  normalized_distance_error - Distance between the center of the predicted
  bounding box and the center of the ground truth bounding box, normalized by
  the ground truth box radius.
  relative_box_size - Ratio of predicted box size to ground truth box size
  (using the diagonal length as the size metric).

  Args:
    prediction: A dictionary with the prediction coordinates.
    ground_truth: A dictionary with the ground truth coordinates.

  Returns:
    A dictionary with the normalized distance error between the predicted and
    ground truth box centers and the relative size of the predicted box.
  """

  center_ground_truth = center_of_bbox(ground_truth)
  center_prediction = center_of_bbox(prediction)
  center_error_km = compute_center_error_km(
      center_prediction, center_ground_truth
  )

  ground_truth_box_size_km = compute_box_size_km(ground_truth)
  normalized_distance_error = center_error_km / ground_truth_box_size_km

  prediction_box_size_km = compute_box_size_km(prediction)
  relative_size = prediction_box_size_km / ground_truth_box_size_km

  return {
      "normalized_distance_error": normalized_distance_error,
      "relative_box_size": relative_size,
  }

def coords_to_box(coords: dict[str, float]) -> np.ndarray:
  return np.array([coords["W"], coords["S"], coords["E"], coords["N"]])


def parse_biodiversity_response(model_response: str) -> dict[str, float]:
  """Parses a model response string into a dictionary of coordinates.

  Args:
    model_response: The model response string to be parsed.

  Returns:
    A dictionary containing the W, E, S, N values.

  Raises:
    ValueError: If the model response string cannot be parsed into either of the
      supported formats.
  """
  if "{" in model_response and "}" in model_response:
    cleaned_response = model_response[
        model_response.find("{") : model_response.rfind("}") + 1
    ]
    return json5.loads(cleaned_response)
  if all(key in model_response for key in ["W", "E", "S", "N"]):
    return {
        "W": float(model_response.split('"W":')[-1].split(",")[0].split()[0]),
        "E": float(model_response.split('"E":')[-1].split(",")[0].split()[0]),
        "S": float(model_response.split('"S":')[-1].split(",")[0].split()[0]),
        "N": float(model_response.split('"N":')[-1].split(",")[0].split()[0]),
    }
  raise ValueError("Can not parse model response")


def bb_intersection_over_union(box_a: np.ndarray, box_b: np.ndarray) -> float:
  """Calculates the Intersection over Union (IoU) between two bounding boxes.

  Args:
    box_a: A list of coordinates representing the first bounding box.
    box_b: A list of coordinates representing the second bounding box.

  Returns:
    The IoU value, a float between 0 and 1.
    0 indicates no overlap and 1 indicates perfect overlap.
  """

  def _intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    width = x_b - x_a
    height = y_b - y_a
    if (width < 0) or (height < 0):
      return 0.0
    return width * height

  def _area(box: np.ndarray) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])

  inter_area = _intersection_area(box_a, box_b)
  union_area = _area(box_a) + _area(box_b) - inter_area
  return inter_area / float(union_area)


def biodiversity_georeferencing_eval(
    model_response: str, ground_truth: str, verbosity: int = 0
) -> dict[str, Union[float, str]]:
  """Computes IOU between ground truth and model response bounding boxes.

  Args:
    ground_truth: A JSON string with lat/lng bounding box coordinates
    model_response: A JSON string with the model response.
    verbosity: Used for debugging.

  Returns:
    A dictionary with IOU keyed as "iou".
  """
  # Load in ground truth coordinates.
  ground_truth_coords = json5.loads(ground_truth)

  try:
    predicted_coords = parse_biodiversity_response(model_response)
  except Exception:  # pylint: disable=broad-except
    if verbosity > 0:
      print("Failed to extract coords from model response: ", model_response)
    return {"iou": "Can not parse model response"}

  iou = bb_intersection_over_union(
      coords_to_box(ground_truth_coords), coords_to_box(predicted_coords)
  )

  distance_metrics = compute_distance_metrics(
      predicted_coords, ground_truth_coords
  )
  biogr_metrics = {
      "iou": iou,
      "normalized_distance_error": distance_metrics[
          "normalized_distance_error"
      ],
      "relative_box_size": distance_metrics["relative_box_size"],
  }
  return biogr_metrics


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
    with open(ground_truth_path, "r",encoding="utf-8") as f:
      ground_truth_info = f.read()

    model_response = ""
    inf_prompt = ""
    if os.path.exists(model_response_path):
      with open(model_response_path, "r",encoding="utf-8") as f:
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


# %%
import concurrent.futures

inference_path=os.path.join(file_root_path, "inference","multi_runs\current\mpve\mat_paper_to_property_1_shot_exclude_trivia",INFER_MODEL_NAME,"run_2\success")
# print(os.listdir(inference_path))
# 获得去掉后缀的文件名
paper_ids = [os.path.splitext(file)[0] for file in os.listdir(inference_path)]
print(paper_ids)
ground_truth_path = os.path.join(file_root_path, "data", "mpve", "ground_truth")
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
    ground_truth_info = preprocess_ground_truth(ground_truth_info, "mpve", "mat_paper_to_property_1_shot_exclude_trivia")
  except Exception as e:
    print(f"ERROR: {e}")
    return None
  if (not "mpve" == 'pdb' and model_response) or ("mpve" == 'pdb' and model_response and inf_prompt):
    res = get_lmsim_score_mpve(model_response, ground_truth_info,paper_id=paper_id)
    return res
  return None
# 使用ThreadPoolExecutor来并行处理文件
with concurrent.futures.ThreadPoolExecutor() as executor:
  results = list(executor.map(process_paper_id, paper_ids))
# 处理结果
for paper_id, result in zip(paper_ids, results):
  if result is not None:
    print(f"Paper ID: {paper_id}, Result: {result}")
  else: 
    print(f"Paper ID: {paper_id} processing failed.")




# %% [markdown]
# ### 保存评估的结果

# %%
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

output_file = os.path.join(inference_path, "..",f"eval_{EVAL_MODEL_NAME} results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# %% [markdown]
# ### 保存平均结果

# %%

f1=0.0
precision=0
recall=0
for data in results:
    f1+=data["f1"]
    precision+=data["precision"]
    recall+=data["recall"]
print(f1/len(results))
print(precision/len(results))
print(recall/len(results))
# Save the results to a JSON file
output_file = os.path.join(inference_path, "..", f"eval_{EVAL_MODEL_NAME} ave_accuracy.json")
with open(output_file, "w") as f:
    json.dump({"f1":f1/len(results),"precision":precision/len(results),"recall":recall/len(results)}, f, indent=4, ensure_ascii=False)
    

