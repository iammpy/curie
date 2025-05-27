import time
import yaml
import requests
import traceback
import json
def call_server(messages,
                model_name,
                model_url
                ):


        # # 提取配置参数
        # if model_name == "DeepSeek-R1-Distill-Qwen-32B":
        #     url = 'http://10.20.4.5:8082/v1/chat/completions'
        # elif model_name == "DeepSeek-R1-Distill-Qwen-7B":
        #     url = 'http://10.20.4.5:8081/v1/chat/completions' 
        # elif model_name== "chem_0320_phy_0324_2to1_math_ckpt_step624_ep2":
        #     url= 'http://10.20.4.2:8002/v1/chat/completions'
        # elif model_name== "chem_0320_phy_0324_2to1_math_add_r1_reasoning_ep1":
        #     url= "http://10.20.4.10:8004/v1/chat/completions"
        # elif model_name == "chemistry_physics_math_7B_16k_rejection_sample_bs256_lr5e-6_roll16_on_aime_gpqa_scibench_global_step_50":
        #     url= "http://10.20.4.14:8006/v1/chat/completions"
        # elif model_name == "our32b_s1math70w_code57w_liucong10w_ch_py_6k_32k":
        #     url = "http://wg-4-11:55320/v1/chat/completions"
        # else:
        #     raise ValueError(f"模型 '{model_name}' 的配置信息未找到。")
        max_retries=3
        # model = "DeepSeek-R1-Distill-Qwen-32B"
        # 重试逻辑
        if model_name == "test":
            import time
            time.sleep(5)
            return "test", "test"
            
        retry_delay
        
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                data_json = {
                    "model": model_name,
                    "messages": [
                       
                        {"role": "user", 
                         "content":messages
                         }
                    ],
                    "temperature": 0.6,
                    "top_p": 0.95
                    }
                response = requests.post(
                    url=model_url,
                    data=json.dumps(data_json),
                    headers = {'Content-Type': 'application/json'}
                    )
                # response.raise_for_status()  # 捕捉非 2xx 状态码
                response_json = response.json()

                # 检查响应格式
                # print(response_json)
                # 返回思考过程和content
                choice = response_json["choices"][0]
                finish_reason = choice["finish_reason"]
                reasoning_content = choice["message"].get("reasoning_content", None)
                content = choice["message"].get("content", None)
                # if reasoning_content is not None:
                #     print(f"思考过程: {reasoning_content}")
                # else:
                #     print("没有返回思考过程。")
                if reasoning_content is None:
                    # 思考过程在content中，用</think> 分割,开头没有<think>
                    # print(content)
                    # reasoning_content = content.split("<think>")[1].split("</think>")[0]
                    # content = content.split("</think>")[1]
                    # content = content.strip()
                    # reasoning_content = reasoning_content.strip()
                    reasoning_content = content.split("</think>")[0]
                    content = content.split("</think>")[1]
                    
                return reasoning_content, content

                
            except Exception as e:
                traceback.print_exc()
                if attempt >= max_retries:
                    print(f"[Warning] get_llm_result_r1_full failed after {max_retries} attempts: {e}")
                    return None
                print(f"第 {attempt} 次调用失败：{e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
                
def call_huoshan(messages, model_name="deepseek-r1", config_path="../colabs/api_config.yaml"):
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
                
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from openai import OpenAI
import os
from typing import Any, Optional, List



    
def call_openai(
    messages,
    client: OpenAI
):

    # # 读取配置文件
    # config_file=os.path.join(os.path.dirname(__file__), "api_config.yaml")
    # # config_file = "api_config.yaml"
    # api_config = yaml.safe_load(open(config_file, "r", encoding="utf-8"))
    model_name = "gpt-4o-2024-11-20"
    # model_cfg = api_config.get(model_name)

    # client = OpenAI(
    #         api_key= api_config.get(model_name, {}).get("api_key"),
    #         base_url= api_config.get(model_name, {}).get("base_url"),
    #         max_retries= api_config.get(model_name, {}).get("max_retries"),
    #         timeout= api_config.get(model_name, {}).get("timeout"),
    #     )
    
    # 测试调用
    # messages = [
    #     {'role': 'user', 'content': '你是谁'}
    # ]
    # completion = client.chat.completions.create(
    #             model=model_name,
    #             messages=messages,
    #             temperature=0.6,
    #             max_completion_tokens=16000,
    #             top_p=0.95,
    #             # seed=seed
    #         )
    # completion_data = completion.model_dump()
    # choice = completion_data.get("choices", [{}])[0]
    # finish_reason = choice.get("finish_reason")
    # response_content = choice.get("message", {}).get("content")
    # # print(f"response: {response_content}")
    # print(f"completion_data: {completion_data}")
    # return finish_reason,response_content
    max_retries = 3
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
            max_completion_tokens=16000,
            top_p=0.95,
            # seed=seed
        )
            completion_data = completion.model_dump()
            choice = completion_data.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason")
            response_content = choice.get("message", {}).get("content")
            # print(f"response: {response_content}")
            # print(f"completion_data: {completion_data}")
            # print("response_content: ", response_content)
            return finish_reason,response_content
            

        except Exception as e:
            traceback.print_exc()
            if attempt >= max_retries:
                print(f"[Warning] get_llm_result_r1_full failed after {max_retries} attempts: {e}")
                return None
            print(f"第 {attempt} 次调用失败：{e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数退避

if __name__ =="__main__":
    res= call_server("你是谁", model_name="our32b_s1math70w_code57w_liucong10w_ch_py_6k_32k")
    print(res)
 
