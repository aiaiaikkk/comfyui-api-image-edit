#!/usr/bin/env python3
"""
ComfyUI API Image Edit Node - Enhanced Version
支持多种API提供商和前端动态UI的图片编辑功能
"""

import io
import json
import base64
import requests
import time
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import torch

class APIImageEditNode:
    """增强版API图片编辑节点"""
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "API/Image Edit"
    
    @classmethod
    def get_provider_models(cls):
        """获取按提供商分类的图像模型字典"""
        return {
            "ModelScope": [
                "Qwen/Qwen-Image-Edit",
                "MusePublic/Qwen-Image-Edit", 
                "Qwen/Qwen-Image",
                "iic/Qwen-Image-Edit",
                "qwen-image-edit",
                "modelscope/stable-diffusion-v1-5-image-edit",
            ],
            "OpenRouter": [
                "google/gemini-2.5-flash-image-preview:free", 
                "anthropic/claude-3-5-sonnet:beta",
                "meta-llama/llama-3.2-90b-vision-instruct:free",
                "qwen/qwen-2-vl-72b-instruct",
                "google/gemini-pro-vision",
            ],
            "OpenAI": [
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-4-vision-preview",
                "dall-e-3",
                "dall-e-2"
            ],
            "Google Gemini": [
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-preview-image-generation", 
                "gemini-2.0-flash-exp-image-generation",
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest",
                "gemini-pro-vision"
            ],
            "PixelWords": [
                "gemini-2.5-flash-image-preview",
                "gemini-2.5-flash-image-preview-hd",
                "gemini-2.5-flash-image",
                "gemini-2.5-flash-image-hd",
                "gpt-4o-image",
                "gpt-4o-dalle", 
                "gpt-4-dalle",
                "gpt-image-1-vip",
                "gpt-4o-image-vip",
                "gpt-4-vision-preview",
                "ideogram",
                "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-3-2b",
                "flux-kontext-max",
                "mj-chat",
                "grok-3-imageGen",
                "seededit",
                "api-images-seededit",
                "gemini-pro-vision",
                "glm-4v",
                "llava-v1.6-34b",
                "playground-v2-1024px-aesthetic"
            ]
        }
    
    @classmethod
    def get_models_for_provider(cls, provider):
        """根据API提供商获取对应的模型列表"""
        provider_models = cls.get_provider_models()
        return provider_models.get(provider, ["No models available"])
    
    @classmethod
    def update_model_list_for_provider(cls, provider):
        """为指定提供商更新模型列表 - ComfyUI前端可调用此方法"""
        models = cls.get_models_for_provider(provider)
        return models
    
    def get_widget_values(self, node_inputs):
        """ComfyUI动态输入支持 - 根据api_provider动态更新model选项"""
        if hasattr(node_inputs, 'get'):
            api_provider = node_inputs.get('api_provider', 'ModelScope')
            if api_provider:
                # 获取该提供商的模型列表
                models = self.get_models_for_provider(api_provider)
                return {"model": models}
        return {}
        
    @classmethod
    def filter_image_models(cls, all_models):
        """筛选图像相关模型"""
        image_keywords = [
            'image', 'vision', 'visual', 'edit', 'dall-e', 'stable-diffusion',
            'qwen-image', 'gemini-pro-vision', 'claude-3', 'gpt-4o'
        ]
        
        filtered_models = []
        for model in all_models:
            model_lower = model.lower()
            if any(keyword in model_lower for keyword in image_keywords):
                filtered_models.append(model)
                
        return filtered_models if filtered_models else all_models
    
    @classmethod
    def INPUT_TYPES(cls):
        provider_names = [
            "ModelScope",
            "OpenRouter", 
            "OpenAI",
            "Google Gemini",
            "PixelWords"
        ]
        
        # 获取所有提供商的模型列表，防止验证错误
        all_provider_models = cls.get_provider_models()
        all_models = []
        
        # 添加所有模型，按提供商分组
        for provider, models in all_provider_models.items():
            all_models.append(f"--- {provider} ---")
            all_models.extend(models)
        
        return {
            "required": {
                "api_provider": (provider_names, {"default": "ModelScope"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "输入API密钥/访问令牌..."}),
                "model": (all_models, {"default": "Qwen/Qwen-Image-Edit"}),
                "prompt": ("STRING", {"default": "Generate or edit images based on the provided inputs", "multiline": True}),
                "refresh_models": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    def __init__(self):
        self.api_configs = {
            "ModelScope": {
                "provider_key": "modelscope",
                "base_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc",
                "edit_endpoint": "/multimodal-generation/generation",
                "type": "dashscope"
            },
            "OpenRouter": {
                "provider_key": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "edit_endpoint": "/chat/completions",
                "type": "openai_compatible"
            },
            "OpenAI": {
                "provider_key": "openai",
                "base_url": "https://api.openai.com/v1",
                "edit_endpoint": "/chat/completions",
                "type": "openai_compatible"
            },
            "Google Gemini": {
                "provider_key": "gemini",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
                "edit_endpoint": ":generateContent",
                "type": "gemini"
            },
            "PixelWords": {
                "provider_key": "pixelwords",
                "base_url": "https://api.sydney-ai.com/v1",
                "edit_endpoint": "/chat/completions",
                "type": "openai_compatible"
            }
        }
        self._model_cache = {}
    
    def tensor_to_pil(self, tensor):
        """将tensor转换为PIL图像"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        tensor = (tensor * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.cpu().numpy())
    
    def pil_to_tensor(self, pil_image):
        """将PIL图像转换为tensor"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        return tensor
    
    def image_to_base64(self, pil_image):
        """将PIL图像转换为base64字符串"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def base64_to_image(self, base64_str):
        """将base64字符串转换为PIL图像"""
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            print(f"Error converting base64 to image: {e}")
            return None
    
    def _download_image_from_url(self, image_url: str) -> Optional[str]:
        """从URL下载图像并转换为base64"""
        try:
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print(f"Error downloading image from URL: {e}")
        return None
    
    def get_provider_key(self, provider_name: str) -> str:
        """从显示名称获取provider key"""
        config = self.api_configs.get(provider_name)
        if config:
            return config.get("provider_key", "modelscope")
        return "modelscope"
    
    def call_dashscope_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str, 
                        mask_b64: Optional[str] = None, **kwargs) -> Optional[str]:
        """调用ModelScope API - 基于参考项目的图像编辑实现
        
        注意：ModelScope有内容过滤机制，可能会阻止某些提示词。
        建议：
        1. 使用简单、明确的提示词
        2. 避免敏感词汇
        3. 如果遇到422错误，系统会自动重试简化参数
        """
        
        # ModelScope图像编辑API使用不同的端点
        url = 'https://api-inference.modelscope.cn/v1/images/generations'
        
        # 深度调试API key编码问题
        print(f"[APIImageEdit] DEBUG - API key长度: {len(api_key) if api_key else 0}")
        if api_key:
            # 检查每个字符的编码
            for i, char in enumerate(api_key):
                if ord(char) > 127:
                    print(f"[APIImageEdit] DEBUG - 发现非ASCII字符 位置{i}: '{char}' (Unicode: {ord(char)})")
            
            # 尝试不同的编码方式
            try:
                api_key.encode('ascii')
                print(f"[APIImageEdit] DEBUG - API key通过ASCII编码检查")
                safe_key = api_key
            except UnicodeEncodeError as e:
                print(f"[APIImageEdit] DEBUG - API key包含非ASCII字符: {e}")
                # 清理API key，只保留ASCII字符
                safe_key = ''.join(c for c in api_key if ord(c) < 128)
                print(f"[APIImageEdit] DEBUG - 清理后API key长度: {len(safe_key)}")
        else:
            safe_key = ''
        
        headers = {
            'Authorization': f'Bearer {safe_key}',
            'Content-Type': 'application/json; charset=utf-8',
            'X-ModelScope-Async-Mode': 'true',
            'User-Agent': 'ComfyUI/1.0'
        }
        
        print(f"[APIImageEdit] DEBUG - Headers: {headers}")
        print(f"[APIImageEdit] DEBUG - API Key length: {len(api_key) if api_key else 0}")
        print(f"[APIImageEdit] DEBUG - Prompt: '{prompt}' (length: {len(prompt)})")
        print(f"[APIImageEdit] DEBUG - Model: '{model}'")
        
        # 构建payload，根据是否有图像输入决定格式
        try:
            payload = {
                'model': model,
                'prompt': prompt
            }
            
            # 如果有图像输入，添加图像数据
            if image_b64:
                image_data = f"data:image/jpeg;base64,{image_b64}"
                payload['image'] = image_data
                print(f"[APIImageEdit] 图像编辑模式")
            else:
                print(f"[APIImageEdit] 纯文本生成模式")
            
            # 添加可选参数
            if kwargs.get("negative_prompt"):
                payload['negative_prompt'] = kwargs["negative_prompt"]
                print(f"[APIImageEdit] 负向提示词: {kwargs['negative_prompt']}")
                
            if kwargs.get("steps", 20) != 20:
                payload['steps'] = kwargs["steps"]
                print(f"[APIImageEdit] 采样步数: {kwargs['steps']}")
                
            if kwargs.get("guidance_scale", 3.5) != 3.5:
                payload['guidance'] = kwargs["guidance_scale"]
                print(f"[APIImageEdit] 引导系数: {kwargs['guidance_scale']}")
                
            if kwargs.get("seed", -1) != -1:
                payload['seed'] = kwargs["seed"]
                print(f"[APIImageEdit] 随机种子: {kwargs['seed']}")
            
            print(f"[APIImageEdit] 开始编辑图片...")
            print(f"[APIImageEdit] 编辑提示: {prompt}")
            print(f"[APIImageEdit] 使用模型: {model}")
            
            # 详细日志记录payload结构
            payload_copy = payload.copy()
            if 'image' in payload_copy:
                payload_copy['image'] = f"data:image/jpeg;base64,{payload_copy['image'][:50]}..."
            print(f"[APIImageEdit] DEBUG - Payload structure: {payload_copy}")
            
            # 编码payload并记录详细信息
            json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            print(f"[APIImageEdit] DEBUG - JSON data length: {len(json_data)}")
            print(f"[APIImageEdit] DEBUG - Sending request to: {url}")
            
            submission_response = requests.post(
                url,
                data=json_data,
                headers=headers,
                timeout=60
            )
            
            if submission_response.status_code != 200:
                print(f"[APIImageEdit] API请求失败: {submission_response.status_code}")
                print(f"[APIImageEdit] 错误详情: {submission_response.text}")
                
                # Handle content filtering error (422)
                if submission_response.status_code == 422:
                    error_text = submission_response.text
                    if "inappropriate content" in error_text or "content filtering" in error_text:
                        print("[APIImageEdit] 检测到内容过滤错误，尝试使用更简单的参数...")
                        
                        # Retry with minimal parameters (like reference project)
                        simple_payload = {
                            'model': model,
                            'prompt': prompt,
                            'image': image_data
                        }
                        
                        try:
                            retry_headers = headers.copy()
                            retry_response = requests.post(
                                url,
                                data=json.dumps(simple_payload, ensure_ascii=False).encode('utf-8'),
                                headers=retry_headers,
                                timeout=60
                            )
                            
                            if retry_response.status_code == 200:
                                print("[APIImageEdit] 简化参数重试成功")
                                submission_json = retry_response.json()
                            else:
                                print(f"[APIImageEdit] 重试仍然失败: {retry_response.status_code}")
                                return None
                        except Exception as retry_error:
                            print(f"[APIImageEdit] 重试请求异常: {str(retry_error)}")
                            return None
                    else:
                        return None
                else:
                    return None
                
            submission_json = submission_response.json()
            result_image_url = None
            
            # 处理异步任务响应
            if 'task_id' in submission_json:
                task_id = submission_json['task_id']
                print(f"[APIImageEdit] 已提交任务，任务ID: {task_id}，开始轮询...")
                poll_start = time.time()
                max_wait_seconds = 720  # 12分钟超时
                
                while True:
                    task_resp = requests.get(
                        f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                        headers={
                            'Authorization': f'Bearer {api_key}',
                            'X-ModelScope-Task-Type': 'image_generation'
                        },
                        timeout=120
                    )
                    
                    if task_resp.status_code != 200:
                        print(f"[APIImageEdit] 任务查询失败: {task_resp.status_code}")
                        print(f"[APIImageEdit] 错误详情: {task_resp.text}")
                        return None
                        
                    task_data = task_resp.json()
                    status = task_data.get('task_status')
                    
                    if status == 'SUCCEED':
                        output_images = task_data.get('output_images') or []
                        if not output_images:
                            print("[APIImageEdit] 任务成功但未返回图片URL")
                            return None
                        result_image_url = output_images[0]
                        print("[APIImageEdit] 任务完成，开始下载编辑后的图片...")
                        break
                        
                    if status == 'FAILED':
                        error_message = task_data.get('errors', {}).get('message', '未知错误')
                        error_code = task_data.get('errors', {}).get('code', '未知错误码')
                        print(f"[APIImageEdit] 任务失败: 错误码 {error_code}, 错误信息: {error_message}")
                        return None
                        
                    if time.time() - poll_start > max_wait_seconds:
                        print("[APIImageEdit] 任务轮询超时")
                        return None
                        
                    print(f"[APIImageEdit] 任务状态: {status}, 继续等待...")
                    time.sleep(5)
                    
            elif 'images' in submission_json and len(submission_json['images']) > 0:
                result_image_url = submission_json['images'][0]['url']
                print("[APIImageEdit] 直接获取到图片URL")
            else:
                print(f"[APIImageEdit] 未识别的API返回格式: {submission_json}")
                return None
            
            # 下载图片
            img_response = requests.get(result_image_url, timeout=30)
            if img_response.status_code != 200:
                print(f"[APIImageEdit] 图片下载失败: {img_response.status_code}")
                return None
                
            # 转换为base64
            import base64
            result_b64 = base64.b64encode(img_response.content).decode('utf-8')
            print("[APIImageEdit] 图片编辑完成！")
            return result_b64
                
        except Exception as e:
            print(f"[APIImageEdit] ModelScope API调用失败: {e}")
            import traceback
            traceback.print_exc()
            
            
            return None
    
    def call_openai_compatible_api(self, provider_name: str, image_b64: Optional[str], prompt: str, model: str, api_key: str,
                                 mask_b64: Optional[str] = None, **kwargs) -> Optional[str]:
        """调用OpenAI兼容的API"""
        config = self.api_configs.get(provider_name)
        if not config:
            return None
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        # 为OpenRouter添加特殊的header
        if config.get("provider_key") == "openrouter":
            headers.update({
                "HTTP-Referer": "https://comfyui.com",
                "X-Title": "ComfyUI API Image Edit"
            })
        
        content = []
        
        # 根据是否有图像输入构建content
        if image_b64:
            # 图像编辑模式
            content = [
                {
                    "type": "text",
                    "text": f"Please edit this image according to the following instructions: {prompt}. Generate a new edited version of the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        else:
            # 纯文本生成模式
            content = [
                {
                    "type": "text",
                    "text": f"Please generate an image according to the following description: {prompt}."
                }
            ]
        
        if mask_b64:
            content.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{mask_b64}"
                }
            })
            content[0]["text"] += " Use the second image as a mask to guide the editing."
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        try:
            url = config["base_url"] + config["edit_endpoint"]
            # 使用UTF-8编码避免中文字符问题
            json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            headers_copy = headers.copy()
            response = requests.post(url, data=json_data, headers=headers_copy, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    
                    # 检查是否有images字段（Gemini格式）
                    if "images" in message and message["images"]:
                        for img_data in message["images"]:
                            if "image_url" in img_data:
                                image_url = img_data["image_url"]["url"]
                                if image_url.startswith("data:image/"):
                                    # 提取base64数据
                                    base64_data = image_url.split(",", 1)[1]
                                    print(f"[APIImageEdit] 从 {provider_name} 获取到图片数据")
                                    return base64_data
                                else:
                                    # 下载URL图片
                                    print(f"[APIImageEdit] 从URL下载图片: {image_url}")
                                    return self._download_image_from_url(image_url)
                    
                    # 检查content字段中的图像数据
                    if "content" in message:
                        content_text = message["content"]
                        print(f"[APIImageEdit] {provider_name} 响应: {content_text[:200]}...")
                        
                        # 尝试查找返回的图像数据
                        import re
                        
                        # 优先查找base64图像数据
                        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                        matches = re.findall(base64_pattern, content_text)
                        
                        if matches:
                            print(f"[APIImageEdit] 从内容中提取base64图片数据")
                            return matches[0]
                        
                        # 查找markdown格式的图像URL 
                        markdown_pattern = r'!\[.*?\]\((https?://[^\s\)]+\.(jpg|jpeg|png|webp|gif))\)'
                        url_matches = re.findall(markdown_pattern, content_text)
                        
                        if url_matches:
                            image_url = url_matches[0][0]  # 取第一个匹配的URL
                            print(f"[APIImageEdit] 从markdown中提取图像URL: {image_url}")
                            return self._download_image_from_url(image_url)
                        
                        # 查找普通的图像URL
                        url_pattern = r'https?://[^\s]+\.(jpg|jpeg|png|webp|gif)'
                        simple_url_matches = re.findall(url_pattern, content_text)
                        
                        if simple_url_matches:
                            # 重建完整URL
                            for match in simple_url_matches:
                                # 在原文本中找到完整URL
                                url_start = content_text.find(f"https://")
                                if url_start != -1:
                                    # 找到URL的结束位置
                                    url_end = content_text.find(" ", url_start)
                                    if url_end == -1:
                                        url_end = content_text.find(")", url_start)
                                    if url_end == -1:
                                        url_end = len(content_text)
                                    
                                    image_url = content_text[url_start:url_end].strip()
                                    print(f"[APIImageEdit] 提取到图像URL: {image_url}")
                                    return self._download_image_from_url(image_url)
                    
                    # 大多数Vision模型主要用于理解而非生成
                    print(f"[APIImageEdit] 警告: {provider_name} Vision模型通常用于理解图像而非生成新图像")
                    return None
                            
            else:
                print(f"{provider_name} API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error calling {provider_name} API: {e}")
        
        return None
    
    def call_claude_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str,
                       mask_b64: Optional[str] = None, **kwargs) -> Optional[str]:
        """调用Claude API"""
        config = self.api_configs["Claude (Anthropic)"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64
                }
            },
            {
                "type": "text",
                "text": f"Please edit this image according to the following instructions: {prompt}. Generate a new edited version."
            }
        ]
        
        payload = {
            "model": model,
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        try:
            url = config["base_url"] + config["edit_endpoint"]
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                
                if "content" in data and data["content"]:
                    content_text = data["content"][0].get("text", "")
                    print(f"Claude response: {content_text[:200]}...")
                    
                    # Claude主要用于理解而非生成图像
                    print("Warning: Claude models typically analyze images rather than generate new ones")
                    return None
                    
            else:
                print(f"Claude API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error calling Claude API: {e}")
        
        return None
    
    def call_gemini_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str,
                       mask_b64: Optional[str] = None, **kwargs) -> Optional[str]:
        """调用Gemini API - 使用官方google-genai库格式"""
        try:
            # 使用官方google-genai库 - 按照文档格式
            from google import genai
            from google.genai import types
            import base64
            
            # 初始化客户端 - 按照官方文档
            client = genai.Client(api_key=api_key.strip())
            
            # 构建输入内容 - 根据是否有图像输入
            contents = []
            
            if image_b64:
                # 图像编辑模式
                image_data = base64.b64decode(image_b64)
                contents = [
                    types.Part.from_text(text=f"Edit this image: {prompt}"),
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                ]
            else:
                # 纯文本生成模式
                contents = [
                    types.Part.from_text(text=f"Generate an image: {prompt}")
                ]
            
            print(f"[APIImageEdit] 调用Gemini API (google-genai库): {model}")
            
            # 调用API - 简化版本，让SDK处理默认配置
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            
            print(f"[APIImageEdit] Gemini API响应成功")
            
            # 处理响应 - 检查文本和图像
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # 检查图像数据
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print("[APIImageEdit] 从Gemini获取到编辑后的图片")
                        return base64.b64encode(part.inline_data.data).decode('utf-8')
                    # 检查文本响应
                    elif hasattr(part, 'text') and part.text:
                        print(f"[APIImageEdit] Gemini响应文本: {part.text[:200]}...")
            
            # 如果没有图像输出，尝试获取响应文本
            if hasattr(response, 'text') and response.text:
                print(f"[APIImageEdit] Gemini完整文本响应: {response.text}")
            
            print("[APIImageEdit] 警告: Gemini响应中未找到图像数据")
            return None
            
        except ImportError:
            print("[APIImageEdit] 错误: 需要安装google-genai库: pip install google-genai")
            # 回退到原来的REST API方式
            return self._call_gemini_rest_api(image_b64, prompt, model, api_key, mask_b64, **kwargs)
        except Exception as e:
            print(f"[APIImageEdit] Gemini API调用异常: {str(e)}")
            # 如果google-genai库调用失败，尝试REST API
            print("[APIImageEdit] 尝试回退到REST API方式...")
            return self._call_gemini_rest_api(image_b64, prompt, model, api_key, mask_b64, **kwargs)
    
    def _call_gemini_rest_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str,
                             mask_b64: Optional[str] = None, **kwargs) -> Optional[str]:
        """Gemini REST API回退方法"""
        config = self.api_configs["Google Gemini"]
        
        # 构建正确的Gemini API URL - 按照官方文档
        url = f"{config['base_url']}/{model}{config['edit_endpoint']}"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key.strip()
        }
        
        # 按照官方文档格式构建请求 - 根据是否有图像输入
        parts = []
        
        if image_b64:
            # 图像编辑模式
            parts = [
                {
                    "text": f"Edit this image according to the following instructions: {prompt}. Generate an edited version of the image."
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg", 
                        "data": image_b64
                    }
                }
            ]
        else:
            # 纯文本生成模式
            parts = [
                {
                    "text": f"Generate an image according to the following description: {prompt}."
                }
            ]
        
        # 使用正确的responseModalities配置
        request_data = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": 1.0,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseModalities": ["TEXT", "IMAGE"]  # 关键配置：同时返回文本和图像
            }
        }
        
        try:
            print(f"[APIImageEdit] 调用Gemini REST API (回退方式): {model}")
            response = requests.post(url, headers=headers, json=request_data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[APIImageEdit] Gemini REST API响应成功")
                
                # 解析Gemini响应 - 按照官方文档格式
                if "candidates" in result and result["candidates"]:
                    candidate = result["candidates"][0]
                    
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            # 检查内联图像数据（主要的图像返回方式）
                            if "inlineData" in part:
                                inline_data = part["inlineData"]
                                if "data" in inline_data:
                                    print("[APIImageEdit] 从Gemini获取到编辑后的图片（inlineData）")
                                    return inline_data["data"]
                            
                            # 检查旧格式的图像数据
                            if "inline_data" in part:
                                inline_data = part["inline_data"]
                                if "data" in inline_data:
                                    print("[APIImageEdit] 从Gemini获取到编辑后的图片（inline_data）")
                                    return inline_data["data"]
                            
                            # 提取文本响应（用于调试）
                            if "text" in part:
                                text_content = part["text"]
                                print(f"[APIImageEdit] Gemini文本响应: {text_content[:200]}...")
                
                # 打印完整响应用于调试
                print(f"[APIImageEdit] DEBUG - 完整Gemini响应: {result}")
                print("[APIImageEdit] Gemini未返回编辑后的图片")
                return None
                        
            else:
                print(f"[APIImageEdit] Gemini REST API错误: {response.status_code}")
                print(f"[APIImageEdit] 错误详情: {response.text}")
                return None
                
        except Exception as e:
            print(f"[APIImageEdit] 调用Gemini REST API异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def edit_image(self, api_provider, api_key, model, prompt, refresh_models=False, 
                  image1=None, image2=None, image3=None, image4=None, mask=None, 
                  strength=1.0, guidance_scale=7.5, steps=20, seed=-1, 
                  watermark=False, negative_prompt=""):
        """主要的图像编辑函数"""
        
        if not api_key or not api_key.strip():
            print("Error: API key is required")
            # 创建一个默认的黑色图像作为错误返回
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image,)
        
        # 验证选择的模型是否属于当前API提供商
        valid_models = self.get_models_for_provider(api_provider)
        # 跳过分隔符行
        if model.startswith("---") and model.endswith("---"):
            print(f"[APIImageEdit] 错误: 请选择具体的模型，而不是分隔符 '{model}'")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image,)
        
        if model not in valid_models:
            print(f"[APIImageEdit] 警告: 模型 '{model}' 不属于 {api_provider} 提供商")
            print(f"[APIImageEdit] {api_provider} 可用模型: {', '.join(valid_models)}")
            print(f"[APIImageEdit] 请在界面中选择正确的模型")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image,)
        
        # 模型列表由前端管理，这里只做基本检查
        if not model.strip():
            # 如果模型为空，使用默认模型
            config = self.api_configs.get(api_provider)
            if config:
                provider_key = config.get("provider_key", "modelscope")
                if provider_key == "modelscope":
                    model = "qwen-max"
                elif provider_key == "openrouter":
                    model = "google/gemini-2.5-flash-image-preview:free"
                elif provider_key == "openai":
                    model = "gpt-4o"
                elif provider_key == "gemini":
                    model = "gemini-2.0-flash-exp"
                else:
                    model = "qwen-max"
                print(f"Using default model: {model}")
            else:
                model = "qwen-max"
                print(f"Fallback to default model: {model}")
        
        # 处理多个可选图像输入
        images = [img for img in [image1, image2, image3, image4] if img is not None]
        
        # 判断生成模式
        if not images:
            # 纯文本生成模式
            print(f"[APIImageEdit] 纯文本生成模式: {prompt}")
            mode = "text_to_image"
            image_b64 = None
        elif len(images) == 1:
            # 单图编辑模式
            print(f"[APIImageEdit] 单图编辑模式")
            mode = "image_to_image"
            pil_image = self.tensor_to_pil(images[0])
            image_b64 = self.image_to_base64(pil_image)
        else:
            # 多图合成模式
            print(f"[APIImageEdit] 多图合成模式，输入图像数量: {len(images)}")
            mode = "multi_image"
            # 处理多张图像，这里使用第一张作为主图，其他作为参考
            pil_image = self.tensor_to_pil(images[0])
            image_b64 = self.image_to_base64(pil_image)
            
            # 增强提示词以包含多图信息
            multi_image_prompt = f"Based on the provided {len(images)} images, {prompt}"
            prompt = multi_image_prompt
        
        mask_b64 = None
        if mask is not None:
            mask_array = mask.cpu().numpy()
            if mask_array.ndim == 3:
                mask_array = mask_array[0]
            
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.convert('RGB')
            mask_b64 = self.image_to_base64(mask_pil)
        
        kwargs = {
            "mask_b64": mask_b64,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed if seed >= 0 else None,
            "watermark": watermark,
            "negative_prompt": negative_prompt
        }
        
        print(f"[APIImageEdit] 调用 {api_provider} API，模型: {model}")
        print(f"[APIImageEdit] 提示词: {prompt[:100]}...")
        
        config = self.api_configs.get(api_provider)
        if not config:
            print(f"Unsupported API provider: {api_provider}")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image,)
        
        api_type = config.get("type", "unknown")
        print(f"[APIImageEdit] API类型: {api_type}")
        
        if api_type == "dashscope":
            result_b64 = self.call_dashscope_api(image_b64, prompt, model, api_key, **kwargs)
        elif api_type == "openai_compatible":
            result_b64 = self.call_openai_compatible_api(api_provider, image_b64, prompt, model, api_key, **kwargs)
        elif api_type == "claude":
            result_b64 = self.call_claude_api(image_b64, prompt, model, api_key, **kwargs)
        elif api_type == "gemini":
            result_b64 = self.call_gemini_api(image_b64, prompt, model, api_key, **kwargs)
        else:
            print(f"Unsupported API type: {api_type}")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image,)
        
        if result_b64:
            try:
                print(f"[APIImageEdit] 接收到base64图片数据，长度: {len(result_b64)}")
                result_image = self.base64_to_image(result_b64)
                if result_image:
                    result_tensor = self.pil_to_tensor(result_image)
                    print("[APIImageEdit] 图像编辑完成，成功返回结果")
                    return (result_tensor,)
                else:
                    print("[APIImageEdit] 错误：无法将base64转换为图像")
            except Exception as e:
                print(f"[APIImageEdit] 处理返回图像时出错: {e}")
        else:
            print("[APIImageEdit] 错误：API调用未返回图像数据")
        
        # 如果API调用失败，根据模式返回适当的图像
        if mode == "text_to_image":
            print("[APIImageEdit] 文本生成失败，返回默认图像")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image,)
        else:
            print("[APIImageEdit] 图像编辑失败，返回原始图像")
            return (images[0],)

NODE_CLASS_MAPPINGS = {
    "APIImageEditNode": APIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIImageEditNode": "API Image Edit (Enhanced)"
}