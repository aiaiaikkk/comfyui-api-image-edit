#!/usr/bin/env python3
"""
ComfyUI API Image Edit Node - Enhanced Version
支持多种API提供商和前端动态UI的图片编辑功能
更新：2025-09-01 16:14 - 修复chat模式API调用问题
"""

import io
import json
import base64
import requests
import time
import os
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import torch

class APIImageEditNode:
    """增强版API图片编辑节点"""
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "response_text")
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
                "black-forest-labs/flux-1-schnell:free",
                "black-forest-labs/flux-1-dev",
                "black-forest-labs/flux-1-pro", 
                "stability-ai/stable-diffusion-3-5-large",
                "openai/dall-e-3",
                "ideogram-ai/ideogram-v2",
            ],
            "OpenAI": [
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-4-vision-preview",
                "dall-e-3",
                "dall-e-3-hd",
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
                "gpt-4o-image-async",
                "gpt-4o-dalle", 
                "gpt-4-dalle",
                "gpt-image-1-vip",
                "gpt-4o-image-vip",
                "gpt-4-vision-preview",
                "dall-e-3-hd",
                "ideogram",
                "ideogram-v2",
                "stable-diffusion",
                "stable-diffusion-xl-1024-v1-0",
                "stable-diffusion-3-2b",
                "stable-diffusion-3.5-large",
                "stable-diffusion-turbo",
                "sd3.5-large",
                "flux-1-schnell",
                "flux-1-dev", 
                "flux-1-pro",
                "flux-kontext-max",
                "flux-kontext-dev",
                "hidream-i1",
                "recraft-v3",
                "leonardo-xl",
                "playground-v2.5",
                "mj-chat",
                "grok-3-imageGen",
                "seededit",
                "api-images-seededit",
                "gemini-pro-vision",
                "glm-4v",
                "llava-v1.6-34b",
                "playground-v2-1024px-aesthetic"
            ],
            "Custom": [
                "custom-model-1",
                "custom-model-2", 
                "custom-model-3",
                "gpt-4o",
                "gpt-4o-mini",
                "claude-3-5-sonnet",
                "gemini-2.5-flash",
                "dall-e-3",
                "stable-diffusion-3.5",
                "flux-1-pro"
            ]
        }
    
    @classmethod
    def get_models_for_provider(cls, provider):
        """根据API提供商获取对应的模型列表"""
        provider_models = cls.get_provider_models()
        return provider_models.get(provider, ["No models available"])
    
    @classmethod
    def is_image_generation_model(cls, provider, model):
        """检查模型是否支持图像生成"""
        image_gen_keywords = [
            'image-generation', 'image-preview', 'dall-e', 'stable-diffusion',
            'flux', 'ideogram', 'seededit', 'mj-chat'
        ]
        
        # Gemini特殊处理
        if provider == "Google Gemini":
            gemini_image_models = [
                "gemini-2.5-flash-image-preview",
                "gemini-2.0-flash-preview-image-generation", 
                "gemini-2.0-flash-exp-image-generation"
            ]
            return model in gemini_image_models
        
        # OpenRouter特殊处理 - 明确的图像生成模型
        if provider == "OpenRouter":
            openrouter_image_models = [
                "google/gemini-2.5-flash-image-preview:free"
            ]
            return model in openrouter_image_models
        
        # 其他提供商的图像生成模型检测
        model_lower = model.lower()
        return any(keyword in model_lower for keyword in image_gen_keywords)
    
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
    
    def get_or_create_session_id(self):
        """获取或创建会话ID"""
        import uuid
        import time
        
        if not self.session_id:
            self.session_id = f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        return self.session_id
    
    def reset_conversation(self):
        """重置对话会话"""
        if self.session_id:
            if self.session_id in self.conversation_sessions:
                del self.conversation_sessions[self.session_id]
            if self.session_id in self.conversation_history:
                del self.conversation_history[self.session_id]
        self.session_id = None
        print("[APIImageEdit] 对话会话已重置")
    
    def get_conversation_history(self, session_id):
        """获取对话历史"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = {
                'messages': [],
                'images': [],  # 存储每轮的图像
                'created_at': time.time()
            }
        return self.conversation_history[session_id]
    
    def add_to_conversation_history(self, session_id, user_message, model_response, image_b64=None):
        """添加到对话历史"""
        history = self.get_conversation_history(session_id)
        
        # 添加用户消息
        history['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': time.time()
        })
        
        # 添加模型响应
        history['messages'].append({
            'role': 'model', 
            'content': model_response,
            'timestamp': time.time()
        })
        
        # 添加生成的图像
        if image_b64:
            history['images'].append({
                'image_b64': image_b64,
                'prompt': user_message,
                'timestamp': time.time()
            })
        
        # 限制历史长度，避免内存溢出
        max_messages = 20
        if len(history['messages']) > max_messages:
            history['messages'] = history['messages'][-max_messages:]
        
        max_images = 10
        if len(history['images']) > max_images:
            history['images'] = history['images'][-max_images:]
    
    @classmethod
    def INPUT_TYPES(cls):
        provider_names = [
            "ModelScope",
            "OpenRouter", 
            "OpenAI",
            "Google Gemini",
            "PixelWords",
            "Custom"
        ]
        
        # 获取所有提供商的模型列表
        all_provider_models = cls.get_provider_models()
        all_models = []
        
        # 添加所有模型，按提供商分组
        for provider in provider_names:
            provider_models = cls.get_models_for_provider(provider)
            all_models.append(f"--- {provider} ---")
            all_models.extend(provider_models)
        
        return {
            "required": {
                "api_provider": (provider_names, {"default": "ModelScope"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "输入API密钥/访问令牌..."}),
                "model": (all_models, {"default": "Qwen/Qwen-Image-Edit"}),
                "prompt": ("STRING", {"default": "Generate or edit images based on the provided inputs", "multiline": True}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                # 生成模式选择
                "generation_mode": (["single", "multiple", "chat", "edit_history"], {"default": "single"}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 16}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "chat_history": ("STRING", {"default": "", "multiline": True, "placeholder": "选择chat模式时显示聊天历史\n（自动更新，通常无需手动输入）"}),
                "edit_history": ("STRING", {"default": "", "multiline": True, "placeholder": "选择edit_history模式时显示编辑历史\n（自动更新，通常无需手动输入）"}),
                "reset_chat": ("BOOLEAN", {"default": False, "label_on": "重置聊天", "label_off": "保持聊天"}),
                "backup_api_url": ("STRING", {"default": "", "multiline": False, "placeholder": "备用API地址或自定义API地址 (如: https://api.custom-provider.com)"}),
                "custom_model": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义模型名称 (仅当选择Custom提供商时使用)"}),
            }
        }
    
    def __init__(self):
        # 多轮对话支持
        self.conversation_sessions = {}  # 存储每个节点的对话会话
        self.conversation_history = {}   # 存储对话历史和图像
        self.session_id = None          # 当前会话ID
        
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
            },
            "Custom": {
                "provider_key": "custom",
                "base_url": "",  # 将由backup_api_url填充
                "edit_endpoint": "/chat/completions",  # 使用chat/completions端点
                "type": "openai_compatible"  # 默认使用OpenAI兼容格式
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
    
    def call_dashscope_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str, **kwargs) -> Optional[str]:
        """调用ModelScope API - 基于参考项目的图像编辑实现
        
        注意：ModelScope有内容过滤机制，可能会阻止某些提示词。
        建议：
        1. 使用简单、明确的提示词
        2. 避免敏感词汇
        3. 如果遇到422错误，系统会自动重试简化参数
        """
        
        # ModelScope图像编辑API使用不同的端点
        url = 'https://api-inference.modelscope.cn/v1/images/generations'
        
        # 清理API key中的非ASCII字符
        if api_key:
            try:
                api_key.encode('ascii')
                safe_key = api_key
            except UnicodeEncodeError:
                safe_key = ''.join(c for c in api_key if ord(c) < 128)
                print(f"[APIImageEdit] API密钥包含非ASCII字符，已自动清理")
        else:
            safe_key = ''
        
        headers = {
            'Authorization': f'Bearer {safe_key}',
            'Content-Type': 'application/json; charset=utf-8',
            'X-ModelScope-Async-Mode': 'true',
            'User-Agent': 'ComfyUI/1.0'
        }
        
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
                
            if kwargs.get("steps", 20) != 20:
                payload['steps'] = kwargs["steps"]
                print(f"[APIImageEdit] 采样步数: {kwargs['steps']}")
                
                
            if kwargs.get("seed", -1) != -1:
                payload['seed'] = kwargs["seed"]
                print(f"[APIImageEdit] 随机种子: {kwargs['seed']}")
            
            print(f"[APIImageEdit] 开始编辑图片...")
            print(f"[APIImageEdit] 编辑提示: {prompt}")
            print(f"[APIImageEdit] 使用模型: {model}")
            
            # 编码payload
            json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            
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
    
    def _extract_text_description(self, content_text: str) -> str:
        """从API响应文本中提取图片描述"""
        import re
        
        # 尝试提取各种格式的图片描述
        patterns = [
            r'\*\*第[一二三四五六七八九十\d]+张图片[：:]\*\*\s*([^*\n]+)',  # **第一张图片：** 描述
            r'第[一二三四五六七八九十\d]+张图片[：:]([^。\n]+)',  # 第一张图片：描述
            r'\*\*第[一二三四五六七八九十\d]+幕[：:]\*\*\s*([^*\n]+)',  # **第一幕：** 描述
            r'第[一二三四五六七八九十\d]+幕[：:]([^。\n]+)',  # 第一幕：描述
            r'Scene\s+\d+[：:]\s*([^.\n]+)',  # Scene 1: 描述
            r'Image\s+\d+[：:]\s*([^.\n]+)',  # Image 1: 描述
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content_text, re.IGNORECASE)
            if matches:
                # 返回第一个匹配的描述，去除多余空格
                return matches[0].strip()
        
        # 如果没有匹配到特定格式，尝试提取第一段文本
        lines = content_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('*') and not line.startswith('#') and len(line) > 10:
                return line[:200]  # 限制长度
        
        return "生成的图片描述"
    
    def call_openai_compatible_api(self, provider_name: str, image_b64: Optional[str], prompt: str, model: str, api_key: str, return_text_desc=False, **kwargs):
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
        
        # 检查是否是多图模式
        images_b64_list = kwargs.get("images_b64", [])
        mode = kwargs.get("mode", "single")
        
        # 根据是否有图像输入构建content
        if mode == "multi_image" and images_b64_list and len(images_b64_list) > 1:
            # 多图合成模式
            content = [
                {
                    "type": "text",
                    "text": f"Please create a composite image based on these {len(images_b64_list)} input images according to the following instructions: {prompt}. Generate a new image that combines elements from all provided images."
                }
            ]
            # 添加所有图像
            for i, img_b64 in enumerate(images_b64_list):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }
                })
            print(f"[APIImageEdit] OpenAI兼容API: 发送{len(images_b64_list)}张图像进行合成")
        elif image_b64:
            # 单图编辑模式
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
        
        
        # 根据端点类型选择不同的请求格式
        if config["edit_endpoint"] == "/completions":
            # 使用completions格式
            if image_b64:
                payload = {
                    "model": model,
                    "prompt": f"Please edit this image according to the following instructions: {prompt}. Generate a new edited version of the image.",
                    "image": image_b64
                }
            else:
                payload = {
                    "model": model,
                    "prompt": f"Please generate an image according to the following description: {prompt}."
                }
        else:
            # 使用chat/completions格式
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user", 
                        "content": content
                    }
                ]
            }
        
        # 添加seed参数（如果API支持）
        if "seed" in kwargs and kwargs["seed"] is not None:
            payload["seed"] = kwargs["seed"]
            print(f"[APIImageEdit] 设置seed: {kwargs['seed']}")
        
        try:
            url = config["base_url"] + config["edit_endpoint"]
            # 使用session来避免编码问题
            session = requests.Session()
            session.headers.update({'User-Agent': 'ComfyUI/1.0'})
            
            # 手动编码JSON数据以确保UTF-8编码
            json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            
            # 复制headers并确保安全编码
            safe_headers = {}
            for key, value in headers.items():
                try:
                    if key == "Authorization":
                        # 特殊处理Authorization header，确保编码安全
                        if isinstance(value, str):
                            # 检查是否包含圆点符号，如果是则提前报错
                            if '●' in value:
                                print(f"[APIImageEdit] 错误：Authorization header包含无效字符(圆点符号)")
                                raise ValueError("Invalid API key format")
                            # 确保只包含ASCII字符
                            safe_headers[key] = value.encode('ascii', errors='strict').decode('ascii')
                        else:
                            safe_headers[key] = str(value)
                    else:
                        # 其他header的处理
                        if isinstance(value, str):
                            safe_headers[key] = value.encode('ascii', errors='ignore').decode('ascii')
                        else:
                            safe_headers[key] = str(value)
                except (UnicodeEncodeError, UnicodeDecodeError, ValueError) as e:
                    print(f"[APIImageEdit] Header编码错误 {key}: {e}")
                    if key == "Authorization":
                        # Authorization header编码失败，直接返回错误
                        print(f"[APIImageEdit] ❌ API密钥包含无效字符，无法编码")
                        return None
                    safe_headers[key] = str(value)
            
            response = session.post(url, data=json_data, headers=safe_headers, timeout=60)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    print(f"[APIImageEdit] {provider_name} API返回非JSON响应: {e}")
                    print(f"[APIImageEdit] 响应内容类型: {response.headers.get('content-type', 'unknown')}")
                    print(f"[APIImageEdit] 响应前500字符: {response.text[:500]}")
                    return None
                
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
                                    print(f"[APIImageEdit] ✅ 从 {provider_name} 获取到base64图片数据")
                                    print(f"[APIImageEdit] 📊 数据长度: {len(base64_data)} 字符")
                                    return base64_data
                                else:
                                    # 下载URL图片
                                    print(f"[APIImageEdit] 从URL下载图片: {image_url}")
                                    return self._download_image_from_url(image_url)
                    
                    # 检查content字段中的图像数据
                    if "content" in message:
                        content_text = message["content"]
                        print(f"[APIImageEdit] {provider_name} 响应: {content_text[:200]}...")
                        
                        # 提取文本描述（用于多图模式）
                        text_description = self._extract_text_description(content_text)
                        
                        # 尝试查找返回的图像数据
                        import re
                        
                        # 优先查找base64图像数据
                        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                        matches = re.findall(base64_pattern, content_text)
                        
                        if matches:
                            print(f"[APIImageEdit] 从内容中提取base64图片数据")
                            if return_text_desc:
                                return (matches[0], text_description)
                            return matches[0]
                        
                        # 查找markdown格式的图像URL 
                        markdown_pattern = r'!\[.*?\]\((https?://[^\s\)]+\.(jpg|jpeg|png|webp|gif))\)'
                        url_matches = re.findall(markdown_pattern, content_text)
                        
                        if url_matches:
                            image_url = url_matches[0][0]  # 取第一个匹配的URL
                            print(f"[APIImageEdit] 从markdown中提取图像URL: {image_url}")
                            image_b64 = self._download_image_from_url(image_url)
                            if return_text_desc:
                                return (image_b64, text_description)
                            return image_b64
                        
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
                                    print(f"[APIImageEdit] 🔗 API返回图像链接: {image_url}")
                                    print(f"[APIImageEdit] 正在下载图像...")
                                    result = self._download_image_from_url(image_url)
                                    if result:
                                        print(f"[APIImageEdit] ✅ 图像下载成功")
                                    else:
                                        print(f"[APIImageEdit] ❌ 图像下载失败")
                                    return result
                    
                    # 大多数Vision模型主要用于理解而非生成
                    print(f"[APIImageEdit] 警告: {provider_name} Vision模型通常用于理解图像而非生成新图像")
                    return None
                            
            else:
                print(f"[APIImageEdit] {provider_name} API请求失败: HTTP {response.status_code}")
                print(f"[APIImageEdit] 请求URL: {url}")
                print(f"[APIImageEdit] 响应内容类型: {response.headers.get('content-type', 'unknown')}")
                print(f"[APIImageEdit] 错误详情: {response.text[:1000]}")
                
                # 检查是否是特定错误类型
                error_text = response.text.lower()
                
                if response.status_code == 400:
                    # 专门检查地理位置限制
                    if ("location is not supported" in error_text or 
                        "user location" in error_text or 
                        "failed_precondition" in error_text):
                        print(f"[APIImageEdit] 🌍 地理位置限制: 该模型在您所在地区不可用")
                        print(f"[APIImageEdit] 💡 建议: 请尝试使用其他没有地区限制的模型，如:")
                        print(f"[APIImageEdit]     - meta-llama/llama-3.2-90b-vision-instruct:free") 
                        print(f"[APIImageEdit]     - anthropic/claude-3-5-sonnet:beta")
                        print(f"[APIImageEdit]     - OpenAI的gpt-4-vision-preview")
                        return None
                
                if response.status_code == 401:
                    print(f"[APIImageEdit] ❌ 认证失败: 请检查API密钥是否正确")
                elif response.status_code == 403:
                    print(f"[APIImageEdit] ❌ 权限不足: 请检查API密钥权限或账户余额")
                elif response.status_code == 404:
                    print(f"[APIImageEdit] ❌ 端点不存在: 请检查API地址和端点配置是否正确")
                elif response.status_code == 429:
                    print(f"[APIImageEdit] ❌ 请求频率过高: 请稍后重试")
                elif response.status_code >= 500:
                    print(f"[APIImageEdit] ❌ 服务器错误: API服务暂时不可用")
                
                return None
                
        except Exception as e:
            print(f"Error calling {provider_name} API: {e}")
        
        return None
    
    def call_claude_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str, **kwargs) -> Optional[str]:
        """调用Claude API"""
        config = self.api_configs["Claude (Anthropic)"]
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json; charset=utf-8"
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
            # 使用session来避免编码问题
            session = requests.Session()
            session.headers.update({'User-Agent': 'ComfyUI/1.0'})
            
            # 手动编码JSON数据以确保UTF-8编码
            json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            
            # 复制headers并确保安全编码
            safe_headers = {}
            for key, value in headers.items():
                try:
                    # 尝试将header值转换为安全的ASCII格式
                    if isinstance(value, str):
                        safe_headers[key] = value.encode('ascii', errors='ignore').decode('ascii')
                    else:
                        safe_headers[key] = str(value)
                except:
                    safe_headers[key] = str(value)
            
            response = session.post(url, data=json_data, headers=safe_headers, timeout=60)
            
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
                       session_id: Optional[str] = None, **kwargs) -> Optional[str]:
        """调用Gemini API - 支持多轮对话的图像编辑"""
        try:
            # 使用官方google-genai库 - 按照文档格式
            from google import genai
            from google.genai import types
            import base64
            
            # 初始化客户端 - 按照官方文档
            client = genai.Client(api_key=api_key.strip())
            
            print(f"[APIImageEdit] 调用Gemini API (google-genai库): {model}")
            print(f"[APIImageEdit] 会话ID: {session_id}")
            
            # 多轮对话支持
            if session_id:
                # 使用Chat API进行多轮对话
                if session_id not in self.conversation_sessions:
                    # 创建新的聊天会话
                    self.conversation_sessions[session_id] = client.chats.create(model=model)
                    print(f"[APIImageEdit] 创建新的对话会话: {session_id}")
                
                chat_session = self.conversation_sessions[session_id]
                
                # 构建多模态消息
                message_parts = []
                
                # 检查是否是多图模式
                images_b64_list = kwargs.get("images_b64", [])
                mode = kwargs.get("mode", "single")
                
                if mode == "multi_image" and images_b64_list and len(images_b64_list) > 1:
                    # 多图合成模式
                    message_parts = [f"Create a composite image based on these {len(images_b64_list)} input images according to the following instructions: {prompt}. Generate a new image that combines elements from all provided images."]
                    
                    for i, img_b64 in enumerate(images_b64_list):
                        image_data = base64.b64decode(img_b64)
                        
                        # 根据图像数据头部检测MIME类型
                        mime_type = "image/jpeg"  # 默认
                        if image_data.startswith(b'\x89PNG'):
                            mime_type = "image/png"
                        elif image_data.startswith(b'GIF'):
                            mime_type = "image/gif"
                        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
                            mime_type = "image/webp"
                        
                        message_parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))
                    
                    print(f"[APIImageEdit] Gemini API: 发送{len(images_b64_list)}张图像进行合成")
                elif image_b64:
                    # 单图编辑模式 - 智能检测图像格式
                    image_data = base64.b64decode(image_b64)
                    
                    # 根据图像数据头部检测MIME类型
                    mime_type = "image/jpeg"  # 默认
                    if image_data.startswith(b'\x89PNG'):
                        mime_type = "image/png"
                    elif image_data.startswith(b'GIF'):
                        mime_type = "image/gif"
                    elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
                        mime_type = "image/webp"
                    
                    print(f"[APIImageEdit] 检测到图像格式: {mime_type}")
                    message_parts = [
                        f"Edit this image according to these instructions: {prompt}. Please generate a new edited version of the image.",
                        types.Part.from_bytes(data=image_data, mime_type=mime_type)
                    ]
                else:
                    # 纯文本生成模式 - 检查历史中是否有图像
                    history = self.get_conversation_history(session_id)
                    if history['images']:
                        # 使用最后一张图像继续编辑
                        last_image = history['images'][-1]
                        image_data = base64.b64decode(last_image['image_b64'])
                        message_parts = [
                            f"Continue editing the previous image: {prompt}",
                            types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
                        ]
                        print(f"[APIImageEdit] 使用历史图像继续编辑")
                    else:
                        message_parts = [f"Generate an image: {prompt}"]
                
                # 发送消息到聊天会话
                response = chat_session.send_message(message_parts)
                
            else:
                # 单次API调用模式（原有逻辑）
                contents = []
                
                # 检查是否是多图模式
                images_b64_list = kwargs.get("images_b64", [])
                mode = kwargs.get("mode", "single")
                
                if mode == "multi_image" and images_b64_list and len(images_b64_list) > 1:
                    # 多图合成模式
                    contents = [types.Part.from_text(text=f"Create a composite image based on these {len(images_b64_list)} input images according to the following instructions: {prompt}. Generate a new image that combines elements from all provided images.")]
                    
                    for i, img_b64 in enumerate(images_b64_list):
                        image_data = base64.b64decode(img_b64)
                        
                        # 根据图像数据头部检测MIME类型
                        mime_type = "image/jpeg"  # 默认
                        if image_data.startswith(b'\x89PNG'):
                            mime_type = "image/png"
                        elif image_data.startswith(b'GIF'):
                            mime_type = "image/gif"
                        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
                            mime_type = "image/webp"
                        
                        contents.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))
                    
                    print(f"[APIImageEdit] Gemini API (单次调用): 发送{len(images_b64_list)}张图像进行合成")
                elif image_b64:
                    # 单图编辑模式 - 智能检测图像格式
                    image_data = base64.b64decode(image_b64)
                    
                    # 根据图像数据头部检测MIME类型
                    mime_type = "image/jpeg"  # 默认
                    if image_data.startswith(b'\x89PNG'):
                        mime_type = "image/png"
                    elif image_data.startswith(b'GIF'):
                        mime_type = "image/gif"
                    elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
                        mime_type = "image/webp"
                    
                    print(f"[APIImageEdit] 检测到图像格式: {mime_type}")
                    contents = [
                        types.Part.from_text(text=f"Edit this image according to these instructions: {prompt}. Please generate a new edited version of the image."),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    ]
                else:
                    # 纯文本生成模式 - 优化的提示词格式
                    enhanced_prompt = f"Generate a high-quality image based on this description: {prompt}. Please create a detailed and visually appealing image that accurately represents the request."
                    contents = [
                        types.Part.from_text(text=enhanced_prompt)
                    ]
                    print(f"[APIImageEdit] 使用增强提示词进行图像生成")
                
                # 调用API - 使用正确的Gemini参数
                from google.genai import types
                
                # 为图像生成优化的配置 - 参考官方最佳实践
                generation_config = {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.8),
                    "top_k": kwargs.get("top_k", 20),
                    "max_output_tokens": kwargs.get("max_output_tokens", 2048),
                }
                
                # 添加seed支持（如果提供）
                if "seed" in kwargs and kwargs["seed"] is not None:
                    generation_config["seed"] = kwargs["seed"]
                    print(f"[APIImageEdit] Gemini google-genai库设置seed: {kwargs['seed']}")
                
                # 如果用户提供了自定义配置，使用用户配置
                if kwargs.get("gemini_params", {}).get("generation_config"):
                    generation_config.update(kwargs["gemini_params"]["generation_config"])
                
                config = types.GenerateContentConfig(**generation_config)
                print(f"[APIImageEdit] 使用生成配置: temperature={generation_config['temperature']}")
                
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
            
            print(f"[APIImageEdit] Gemini API响应成功")
            
            # 处理响应 - 检查文本和图像
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # 检查图像数据
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print("[APIImageEdit] 从Gemini获取到编辑后的图片")
                        result_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                        
                        # 保存到对话历史
                        if session_id:
                            response_text = getattr(response, 'text', 'Image generated successfully')
                            self.add_to_conversation_history(session_id, prompt, response_text, result_b64)
                            print(f"[APIImageEdit] 对话历史已更新")
                        
                        return result_b64
                    # 检查文本响应
                    elif hasattr(part, 'text') and part.text:
                        print(f"[APIImageEdit] Gemini响应文本: {part.text[:200]}...")
            
            # 如果没有图像输出，尝试获取响应文本
            if hasattr(response, 'text') and response.text:
                print(f"[APIImageEdit] Gemini返回文本响应但无图像输出")
                
                # 即使没有图像输出，也保存对话历史
                if session_id:
                    self.add_to_conversation_history(session_id, prompt, response.text)
                
                # 如果是不支持图像生成的模型，但在聊天模式下，尝试使用最后一张图像
                if session_id:
                    history = self.get_conversation_history(session_id)
                    if history['images'] and image_b64:
                        print(f"[APIImageEdit] 模型不支持图像生成，返回输入图像以保持对话连续性")
                        return image_b64
            
            print("[APIImageEdit] 警告: Gemini响应中未找到图像数据")
            return None
            
        except ImportError:
            print("[APIImageEdit] 错误: 需要安装google-genai库: pip install google-genai")
            # 回退到原来的REST API方式
            return self._call_gemini_rest_api(image_b64, prompt, model, api_key, **kwargs)
        except Exception as e:
            error_message = str(e).lower()
            
            # 根据官方文档处理常见错误
            if "quota exceeded" in error_message or "rate limit" in error_message:
                print("[APIImageEdit] ⚠️ Gemini API配额已用完或达到速率限制，请稍后重试")
            elif "invalid api key" in error_message or "authentication" in error_message:
                print("[APIImageEdit] ❌ Gemini API密钥无效，请检查API密钥")
            elif "model not found" in error_message:
                print(f"[APIImageEdit] ❌ Gemini模型 '{model}' 不存在或不可用")
            elif "safety" in error_message:
                print("[APIImageEdit] ⚠️ 内容被Gemini安全过滤器拦截，请尝试修改提示词")
            else:
                print(f"[APIImageEdit] Gemini API调用异常: {str(e)}")
            
            # 如果google-genai库调用失败，尝试REST API
            print("[APIImageEdit] 尝试回退到REST API方式...")
            return self._call_gemini_rest_api(image_b64, prompt, model, api_key, **kwargs)
    
    def _call_gemini_rest_api(self, image_b64: Optional[str], prompt: str, model: str, api_key: str, **kwargs) -> Optional[str]:
        """Gemini REST API回退方法"""
        config = self.api_configs["Google Gemini"]
        
        # 构建正确的Gemini API URL - 按照官方文档
        url = f"{config['base_url']}/{model}{config['edit_endpoint']}"
        
        headers = {
            "Content-Type": "application/json; charset=utf-8",
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
        
        # 构建生成配置
        generation_config = {
            "temperature": 1.0,
            "topP": 0.95,
            "maxOutputTokens": 8192,
            "responseModalities": ["TEXT", "IMAGE"]  # 关键配置：同时返回文本和图像
        }
        
        # 添加seed支持（如果提供）
        if "seed" in kwargs and kwargs["seed"] is not None:
            generation_config["seed"] = kwargs["seed"]
            print(f"[APIImageEdit] Gemini REST API设置seed: {kwargs['seed']}")
        
        # 使用正确的responseModalities配置
        request_data = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": generation_config
        }
        
        try:
            print(f"[APIImageEdit] 调用Gemini REST API (回退方式): {model}")
            # 使用session来避免编码问题
            session = requests.Session()
            session.headers.update({'User-Agent': 'ComfyUI/1.0'})
            
            # 手动编码JSON数据以确保UTF-8编码
            json_data = json.dumps(request_data, ensure_ascii=False).encode('utf-8')
            
            # 复制headers并确保安全编码
            safe_headers = {}
            for key, value in headers.items():
                try:
                    # 尝试将header值转换为安全的ASCII格式
                    if isinstance(value, str):
                        safe_headers[key] = value.encode('ascii', errors='ignore').decode('ascii')
                    else:
                        safe_headers[key] = str(value)
                except:
                    safe_headers[key] = str(value)
            
            response = session.post(url, data=json_data, headers=safe_headers, timeout=120)
            
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
                            
                            # 提取文本响应
                            if "text" in part:
                                text_content = part["text"]
                                print(f"[APIImageEdit] Gemini文本响应: {text_content[:200]}...")
                
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
    
    def edit_image(self, api_provider, api_key, model, prompt,
                  image1=None, image2=None, image3=None, image4=None,
                  generation_mode="single", image_count=1, seed=-1,
                  chat_history="", edit_history="", reset_chat=False, backup_api_url="", custom_model=""):
        """主要的图像编辑函数 - 支持多轮对话"""
        
        # 处理seed参数 - 支持随机抽卡
        import random
        if seed == -1:
            actual_seed = random.randint(0, 2147483647)
            print(f"[APIImageEdit] 使用随机seed: {actual_seed}")
        else:
            actual_seed = seed
            print(f"[APIImageEdit] 使用固定seed: {actual_seed}")
        
        # 获取或创建会话ID - 在聊天模式下需要会话管理
        session_id = self.get_or_create_session_id() if generation_mode == "chat" else None
        
        # 更新历史记录显示
        if generation_mode == "chat" and session_id:
            history = self.get_conversation_history(session_id)
            
            # 生成聊天历史显示文本
            chat_display = []
            for i, msg in enumerate(history['messages'][-10:]):  # 只显示最近10条
                role_icon = "🤖" if msg['role'] == 'model' else "👤"
                chat_display.append(f"{role_icon} {msg['content'][:100]}...")
            
            # 生成编辑历史显示文本  
            edit_display = []
            for i, img in enumerate(history['images'][-5:]):  # 只显示最近5张
                timestamp = time.strftime("%H:%M:%S", time.localtime(img['timestamp']))
                edit_display.append(f"🎨 {timestamp}: {img['prompt'][:80]}...")
            
            # 更新显示内容
            current_chat_history = "\n".join(chat_display) if chat_display else "暂无聊天记录"
            current_edit_history = "\n".join(edit_display) if edit_display else "暂无编辑记录"
            
            print(f"[APIImageEdit] 当前聊天历史: {len(history['messages'])}条消息")
            print(f"[APIImageEdit] 当前编辑历史: {len(history['images'])}张图像")
        
        if not api_key or not api_key.strip():
            print("Error: API key is required")
            # 创建一个默认的黑色图像作为错误返回
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "API密钥未提供")
            
        # 检查API密钥是否是前端隐藏的圆点符号
        if api_key.strip() and all(c == '●' for c in api_key.strip()):
            print("Error: API key appears to be masked with dots (●). Please enter your real API key.")
            print("提示：看起来您输入的是被隐藏的API密钥。请重新输入真实的API密钥。")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "API密钥格式错误：请输入真实的API密钥，而不是圆点符号")
        
        # 验证选择的模型是否属于当前API提供商
        valid_models = self.get_models_for_provider(api_provider)
        # 跳过分隔符行
        if model.startswith("---") and model.endswith("---"):
            print(f"[APIImageEdit] 错误: 请选择具体的模型，而不是分隔符 '{model}'")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "请选择具体的模型，而不是分隔符")
        
        # 自定义提供商的特殊处理
        if api_provider == "Custom":
            if custom_model.strip():
                # 如果用户提供了自定义模型名称，优先使用
                model = custom_model.strip()
                print(f"[APIImageEdit] 使用用户指定的自定义模型: {model}")
            elif model in valid_models:
                # 如果选择的是预定义的自定义模型列表中的模型，直接使用
                print(f"[APIImageEdit] 使用预定义的自定义模型: {model}")
            else:
                # 如果模型不在列表中，但在自定义提供商模式下，仍然允许使用
                print(f"[APIImageEdit] 自定义提供商模式：允许使用模型 '{model}'")
                print(f"[APIImageEdit] 提示：建议在'自定义模型'字段中明确指定模型名称")
        elif model not in valid_models:
            print(f"[APIImageEdit] 警告: 模型 '{model}' 不属于 {api_provider} 提供商")
            print(f"[APIImageEdit] {api_provider} 可用模型: {', '.join(valid_models)}")
            print(f"[APIImageEdit] 请在界面中选择正确的模型")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "请选择正确的模型")
        
        # 检查模型是否支持图像生成（聊天模式需要）
        supports_image_gen = self.is_image_generation_model(api_provider, model)
        if generation_mode == "chat" and not supports_image_gen:
            print(f"[APIImageEdit] 警告: 模型 '{model}' 不支持图像生成，聊天模式功能可能无法正常工作")
            print(f"[APIImageEdit] 推荐使用图像生成模型如: gemini-2.5-flash-image-preview")
            # 继续执行，但用户会看到警告
        
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
        

        # 处理自定义提供商和备用API地址
        original_config = None
        if api_provider == "Custom":
            if not backup_api_url.strip():
                print("[APIImageEdit] 错误: 自定义提供商需要提供API地址")
                default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (default_image, "自定义提供商需要提供API地址")
            
            # 设置自定义提供商的配置
            original_config = self.api_configs.get("Custom", {}).copy()
            self.api_configs["Custom"]["base_url"] = backup_api_url.strip()
            
            # 处理自定义模型
            if custom_model.strip():
                model = custom_model.strip()
                print(f"[APIImageEdit] 使用自定义模型: {model}")
            else:
                # 如果没有提供自定义模型，使用默认模型
                if model in ["custom-model-1", "custom-model-2", "custom-model-3"]:
                    model = "gpt-4o"  # 默认模型
                    print(f"[APIImageEdit] 使用默认模型: {model}")
                
            print(f"[APIImageEdit] 自定义提供商配置:")
            print(f"[APIImageEdit]   API地址: {backup_api_url}")
            print(f"[APIImageEdit]   模型: {model}")
            
        elif backup_api_url.strip():
            print(f"[APIImageEdit] 使用备用API地址: {backup_api_url}")
            # 临时修改API配置
            original_config = self.api_configs.get(api_provider, {}).copy()
            if original_config and "base_url" in original_config:
                self.api_configs[api_provider]["base_url"] = backup_api_url.strip()
                print(f"[APIImageEdit] 已切换到备用地址: {backup_api_url}")
        
        # 根据generation_mode进行分支处理
        if generation_mode == "multiple":
            # 多图生成模式
            print(f"[APIImageEdit] 多图生成模式：生成{image_count}张图像")
            result = self._handle_multiple_generation(api_provider, api_key, model, prompt, image_count, images, actual_seed)
            
            # 恢复原始API配置
            if original_config:
                self.api_configs[api_provider] = original_config
                print(f"[APIImageEdit] 已恢复原始API配置")
            
            return result
            
        elif generation_mode == "chat":
            # 聊天模式 - 使用chat_history
            print(f"[APIImageEdit] 聊天模式：使用聊天历史")
            result = self._handle_chat_mode(api_provider, api_key, model, prompt, chat_history, images, reset_chat, actual_seed)
            
            # 恢复原始API配置
            if original_config:
                self.api_configs[api_provider] = original_config
                print(f"[APIImageEdit] 已恢复原始API配置")
            
            return result
            
        elif generation_mode == "edit_history":
            # 编辑历史模式
            print(f"[APIImageEdit] 编辑历史模式：使用编辑历史")
            result = self._handle_edit_history_mode(api_provider, api_key, model, prompt, edit_history, images, actual_seed)
            
            # 恢复原始API配置
            if original_config:
                self.api_configs[api_provider] = original_config
                print(f"[APIImageEdit] 已恢复原始API配置")
            
            return result
        
        # 默认单图模式或聊天模式下的图像处理
        if generation_mode == "chat" and session_id:
            history = self.get_conversation_history(session_id)
            
            if history['images'] and not images:
                # 使用历史图像进行对话编辑
                last_image = history['images'][-1]
                image_b64 = last_image['image_b64']
                print(f"[APIImageEdit] 使用历史图像进行对话编辑 (来自: {last_image['prompt'][:50]}...)")
                mode = "conversation_edit"
            elif images:
                # 有新图像输入，使用新图像
                mode = "image_to_image" if len(images) == 1 else "multi_image"
                pil_image = self.tensor_to_pil(images[0])
                image_b64 = self.image_to_base64(pil_image)
                if len(images) > 1:
                    multi_image_prompt = f"Based on the provided {len(images)} images, {prompt}"
                    prompt = multi_image_prompt
                print(f"[APIImageEdit] 对话模式 - 使用新输入图像")
            else:
                # 没有图像输入，也没有历史图像
                if not history['images']:
                    print(f"[APIImageEdit] 对话模式下的纯文本生成: {prompt}")
                    mode = "conversation_text_to_image"
                    image_b64 = None
                else:
                    print(f"[APIImageEdit] 基于文本生成图像")
                    mode = "conversation_text_to_image"
                    image_b64 = None
        else:
            # 原有逻辑：非对话模式或未启用使用历史图像
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
                # 处理多张图像，将所有图像转换为base64
                images_b64 = []
                for i, img in enumerate(images):
                    pil_img = self.tensor_to_pil(img)
                    img_b64 = self.image_to_base64(pil_img)
                    images_b64.append(img_b64)
                    print(f"[APIImageEdit] 已处理第{i+1}张图像")
                
                # 主图像（用于兼容单图API调用）
                image_b64 = images_b64[0]
                
                # 增强提示词以包含多图信息
                multi_image_prompt = f"Based on the provided {len(images)} images, {prompt}"
                prompt = multi_image_prompt
        
        
        # 构建正确的Gemini API参数（移除无效参数）
        gemini_params = {}
        if api_provider == "Google Gemini":
            # Gemini只支持这些参数
            generation_config = {}
            if steps > 20:
                generation_config["maxOutputTokens"] = min(steps * 50, 8192)  # 粗略映射
            gemini_params["generation_config"] = generation_config
        
        # 其他API保持原有参数结构
        kwargs = {
            "gemini_params": gemini_params,
            "seed": actual_seed
        }
        
        # 如果是多图合成模式，添加所有图像信息
        if mode == "multi_image" and 'images_b64' in locals():
            kwargs["images_b64"] = images_b64
            kwargs["mode"] = mode
            print(f"[APIImageEdit] 准备发送{len(images_b64)}张图像到API")
        
        config = self.api_configs.get(api_provider)
        if not config:
            print(f"Unsupported API provider: {api_provider}")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "API密钥未提供")
        
        api_type = config.get("type", "unknown")
        
        print(f"[APIImageEdit] 调用 {api_provider} API，模型: {model}")
        print(f"[APIImageEdit] 🔧 单图生成配置:")
        print(f"[APIImageEdit]    - API提供商: {api_provider}")
        print(f"[APIImageEdit]    - 模型: {model}")
        print(f"[APIImageEdit]    - API类型: {api_type}")
        print(f"[APIImageEdit]    - Seed: {actual_seed}")
        if len(prompt) > 200:
            print(f"[APIImageEdit]    - 提示词: {prompt[:200]}...")
        else:
            print(f"[APIImageEdit]    - 提示词: {prompt}")
        
        if api_type == "dashscope":
            result_b64 = self.call_dashscope_api(image_b64, prompt, model, api_key, **kwargs)
        elif api_type == "openai_compatible":
            result_b64 = self.call_openai_compatible_api(api_provider, image_b64, prompt, model, api_key, **kwargs)
        elif api_type == "claude":
            result_b64 = self.call_claude_api(image_b64, prompt, model, api_key, **kwargs)
        elif api_type == "gemini":
            result_b64 = self.call_gemini_api(image_b64, prompt, model, api_key, 
                                            session_id=session_id, **kwargs)
        else:
            print(f"Unsupported API type: {api_type}")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "API密钥未提供")
        
        if result_b64:
            try:
                print(f"[APIImageEdit] 接收到base64图片数据，长度: {len(result_b64)}")
                result_image = self.base64_to_image(result_b64)
                if result_image:
                    result_tensor = self.pil_to_tensor(result_image)
                    
                    # 构建详细的返回信息
                    success_response = f"✅ 单图生成成功\n\n"
                    success_response += f"🔧 生成配置：\n"
                    success_response += f"   • API提供商: {api_provider}\n"
                    success_response += f"   • 模型: {model}\n"
                    success_response += f"   • API类型: {api_type}\n"
                    success_response += f"   • Seed: {actual_seed}\n\n"
                    success_response += f"📝 使用的完整提示词：\n"
                    if images:
                        actual_prompt = f"Please edit this image according to the following instructions: {prompt}. Generate a new edited version of the image."
                        success_response += f"   原始请求: {prompt}\n"
                        success_response += f"   API实际提示词: {actual_prompt}"
                    else:
                        actual_prompt = f"Please generate an image according to the following description: {prompt}."
                        success_response += f"   原始请求: {prompt}\n"
                        success_response += f"   API实际提示词: {actual_prompt}"
                    
                    print("[APIImageEdit] 图像编辑完成，成功返回结果")
                    
                    # 恢复原始API配置
                    if original_config:
                        self.api_configs[api_provider] = original_config
                        print(f"[APIImageEdit] 已恢复原始API配置")
                    
                    return (result_tensor, success_response)
                else:
                    print("[APIImageEdit] 错误：无法将base64转换为图像")
            except Exception as e:
                print(f"[APIImageEdit] 处理返回图像时出错: {e}")
        else:
            print("[APIImageEdit] 错误：API调用未返回图像数据")
        
        # 如果API调用失败，根据模式返回适当的图像
        # 恢复原始API配置（在错误情况下）
        if original_config:
            self.api_configs[api_provider] = original_config
            print(f"[APIImageEdit] 已恢复原始API配置")
        
        if mode == "text_to_image":
            print("[APIImageEdit] 文本生成失败，返回默认图像")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "文本生成失败")
        else:
            print("[APIImageEdit] 图像编辑失败，返回原始图像")
            return (images[0], "图像编辑失败")
    
    def _handle_multiple_generation(self, api_provider, api_key, model, prompt, image_count, images, actual_seed):
        """处理多图生成模式"""
        # 获取API配置
        config = self.api_configs.get(api_provider)
        if not config:
            print(f"[APIImageEdit] 不支持的API提供商: {api_provider}")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, f"不支持的API提供商: {api_provider}")
        
        api_type = config.get("type", "unknown")
        
        # 检查模型是否适合图像生成
        is_image_gen_model = self.is_image_generation_model(api_provider, model)
        if not is_image_gen_model:
            print(f"[APIImageEdit] 警告: 模型 '{model}' 可能不是专门的图像生成模型")
        else:
            print(f"[APIImageEdit] 使用图像生成模型: {model}")
        
        print(f"[APIImageEdit] 开始生成{image_count}张图像")
        print(f"[APIImageEdit] 🔧 配置信息:")
        print(f"[APIImageEdit]    - API提供商: {api_provider}")
        print(f"[APIImageEdit]    - 模型: {model}")
        print(f"[APIImageEdit]    - 基础提示词: {prompt}")
        print(f"[APIImageEdit]    - 基础seed: {actual_seed}")
        print(f"[APIImageEdit]    - API类型: {api_type}")
        
        generated_images = []
        prompt_details = []  # 存储每张图的详细提示词信息
        
        # 循环生成多张图像
        for i in range(image_count):
            print(f"[APIImageEdit] 正在生成第{i+1}/{image_count}张图像...")
            
            try:
                # 为每张图生成不同的seed（基于base_seed + index）
                import random
                if actual_seed == -1:
                    current_seed = random.randint(0, 2147483647)
                else:
                    current_seed = actual_seed + i  # 基于原始seed的偏移
                
                # 构建包含seed的kwargs
                generation_kwargs = {"seed": current_seed, "return_text_desc": True}
                
                # 根据API类型调用相应的方法，获取图像和文本描述
                result = None
                if api_type == "openai_compatible":
                    result = self.call_openai_compatible_api(api_provider, None, prompt, model, api_key, **generation_kwargs)
                elif api_type == "dashscope":
                    result = self.call_dashscope_api(None, prompt, model, api_key, **generation_kwargs)
                elif api_type == "gemini":
                    result = self.call_gemini_api(None, prompt, model, api_key, **generation_kwargs)
                else:
                    print(f"[APIImageEdit] 不支持的API类型: {api_type}")
                    result = None
                
                # 解析返回结果
                result_b64 = None
                text_description = "生成的图片描述"
                
                if result:
                    if isinstance(result, tuple) and len(result) == 2:
                        result_b64, text_description = result
                    else:
                        result_b64 = result
                        text_description = f"Please generate an image according to the following description: {prompt}."
                
                # 记录详细信息（使用提取的文本描述）
                image_detail = f"图像 #{i+1}: seed={current_seed}, 具体场景=\"{text_description}\""
                prompt_details.append(image_detail)
                print(f"[APIImageEdit]    📝 {image_detail}")
                
                if result_b64:
                    generated_images.append(result_b64)
                    print(f"[APIImageEdit] ✅ 第{i+1}张图像生成成功")
                else:
                    print(f"[APIImageEdit] ❌ 第{i+1}张图像生成失败")
                    
            except Exception as e:
                print(f"[APIImageEdit] 第{i+1}张图像生成异常: {e}")
        
        # 构建详细的返回信息
        detailed_response = f"📊 多图生成完成：成功生成 {len(generated_images)}/{image_count} 张图像\n\n"
        detailed_response += f"🔧 生成配置：\n"
        detailed_response += f"   • API提供商: {api_provider}\n"
        detailed_response += f"   • 模型: {model}\n"
        detailed_response += f"   • API类型: {api_type}\n\n"
        detailed_response += f"📝 详细提示词记录：\n"
        detailed_response += f"   • 基础提示词: {prompt}\n"
        for detail in prompt_details:
            detailed_response += f"   • {detail}\n"
        
        print(f"[APIImageEdit] 📋 生成总结:")
        print(detailed_response)
        
        if generated_images:
            result_tensors = []
            print(f"[APIImageEdit] 开始处理{len(generated_images)}张生成的图像")
            for i, img_b64 in enumerate(generated_images):
                img_pil = self.base64_to_image(img_b64)
                if img_pil:
                    tensor = self.pil_to_tensor(img_pil)
                    result_tensors.append(tensor)
                    print(f"[APIImageEdit] 成功转换第{i+1}张图像为tensor，形状: {tensor.shape}")
                else:
                    print(f"[APIImageEdit] 第{i+1}张图像转换失败")
            
            if result_tensors:
                combined_tensor = torch.cat(result_tensors, dim=0)
                print(f"[APIImageEdit] 合并后的tensor形状: {combined_tensor.shape}")
                return (combined_tensor, detailed_response)
        
        print("[APIImageEdit] 多图生成失败")
        default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return (default_image, "多图生成失败")
    
    def _handle_chat_mode(self, api_provider, api_key, model, prompt, chat_history, images, reset_chat=False, actual_seed=-1):
        """处理聊天模式 - 保持对话连续性"""
        
        # 处理重置聊天
        if reset_chat:
            if hasattr(self, '_auto_chat_history'):
                self._auto_chat_history = ""
            if hasattr(self, '_last_image_b64'):
                self._last_image_b64 = None
            print(f"[APIImageEdit] 聊天历史已重置")
        
        # 自动对话状态管理 - 如果chat_history为空，尝试从类实例中获取
        if not chat_history or not chat_history.strip():
            if not hasattr(self, '_auto_chat_history'):
                self._auto_chat_history = ""
            chat_history = self._auto_chat_history
            print(f"[APIImageEdit] 使用自动管理的聊天历史，长度: {len(chat_history)}")
        
        print(f"[APIImageEdit] 聊天模式启动，历史记录长度: {len(chat_history) if chat_history else 0}")
        
        # 构建包含历史记录的完整提示词
        if chat_history.strip():
            # 解析聊天历史，提取最新的图像信息
            lines = chat_history.strip().split('\n')
            previous_context = "\n".join(lines[-10:]) if len(lines) > 10 else chat_history  # 保留最近10轮对话
            full_prompt = f"Previous conversation context:\n{previous_context}\n\nCurrent request: {prompt}"
            print(f"[APIImageEdit] 🔧 聊天模式配置:")
            print(f"[APIImageEdit]    - API提供商: {api_provider}")
            print(f"[APIImageEdit]    - 模型: {model}")
            print(f"[APIImageEdit]    - Seed: {actual_seed}")
            print(f"[APIImageEdit]    - 历史记录行数: {len(lines)}")
            print(f"[APIImageEdit]    - 原始请求: {prompt}")
            print(f"[APIImageEdit]    - 构建的完整提示词: {full_prompt[:500]}{'...' if len(full_prompt) > 500 else ''}")
        else:
            full_prompt = prompt
            print(f"[APIImageEdit] 首次聊天，无历史记录")
        
        # API最终发送的提示词将在后面根据具体情况构建
        
        # 根据是否有图像输入选择处理方式
        config = self.api_configs.get(api_provider)
        if not config:
            print(f"[APIImageEdit] 不支持的API提供商: {api_provider}")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, f"不支持的API提供商: {api_provider}")
            
        api_type = config.get("type", "unknown")
        
        if images:
            # 有图像输入时，进行图像编辑
            pil_img = self.tensor_to_pil(images[0])
            image_b64 = self.image_to_base64(pil_img)
            print(f"[APIImageEdit] 聊天模式：编辑输入图像")
            operation_type = "图像编辑"
            api_final_prompt = f"Please edit this image according to the following instructions: {full_prompt}. Generate a new edited version of the image."
        else:
            # 检查是否有上一轮的图像可以使用
            if hasattr(self, '_last_image_b64') and self._last_image_b64:
                image_b64 = self._last_image_b64
                print(f"[APIImageEdit] 聊天模式：基于上一轮图像继续编辑")
                operation_type = "图像编辑"
                api_final_prompt = f"Please edit this image according to the following instructions: {full_prompt}. Generate a new edited version of the image."
            else:
                # 纯文本聊天，生成新图像
                print(f"[APIImageEdit] 聊天模式：生成新图像")
                image_b64 = None
                operation_type = "图像生成"
                api_final_prompt = f"Please generate an image according to the following description: {full_prompt}."
        
        # 检查模型是否适合图像生成
        is_image_gen_model = self.is_image_generation_model(api_provider, model)
        if not is_image_gen_model:
            print(f"[APIImageEdit] 警告: 模型 '{model}' 可能不是专门的图像生成模型")
            print(f"[APIImageEdit] 建议使用图像生成模型以获得更好的效果")
        else:
            print(f"[APIImageEdit] 使用图像生成模型: {model}")
        
        # 调用相应的API
        generation_kwargs = {"seed": actual_seed} if actual_seed != -1 else {}
        try:
            if api_type == "openai_compatible":
                result_b64 = self.call_openai_compatible_api(api_provider, image_b64, full_prompt, model, api_key, **generation_kwargs)
            elif api_type == "dashscope":
                result_b64 = self.call_dashscope_api(image_b64, full_prompt, model, api_key, **generation_kwargs)
            elif api_type == "gemini":
                result_b64 = self.call_gemini_api(image_b64, full_prompt, model, api_key, **generation_kwargs)
            else:
                print(f"[APIImageEdit] 不支持的API类型: {api_type}")
                result_b64 = None
        except Exception as e:
            print(f"[APIImageEdit] API调用异常: {e}")
            result_b64 = None
        
        if result_b64:
            result_image = self.base64_to_image(result_b64)
            if result_image:
                result_tensor = self.pil_to_tensor(result_image)
                
                # 构建新的聊天历史记录
                import time
                timestamp = time.strftime("%H:%M:%S")
                new_history_entry = f"[{timestamp}] User: {prompt}\n[{timestamp}] Assistant: 完成{operation_type}"
                
                updated_history = f"{chat_history}\n{new_history_entry}".strip() if chat_history else new_history_entry
                
                # 自动保存聊天历史到类实例中
                self._auto_chat_history = updated_history
                
                # 保存生成的图像base64数据供下一轮使用
                self._last_image_b64 = result_b64
                
                # 构建详细的返回信息
                chat_response = f"💬 聊天模式{operation_type}成功\n\n"
                chat_response += f"🔧 生成配置：\n"
                chat_response += f"   • API提供商: {api_provider}\n"
                chat_response += f"   • 模型: {model}\n"
                chat_response += f"   • API类型: {api_type}\n"
                chat_response += f"   • Seed: {actual_seed}\n"
                chat_response += f"   • 操作类型: {operation_type}\n\n"
                chat_response += f"📝 使用的完整提示词：\n"
                chat_response += f"   • 原始请求: {prompt}\n"
                chat_response += f"   • 构建的上下文提示词: \n{full_prompt}\n"
                chat_response += f"   • API最终发送提示词: \n{api_final_prompt}\n\n"
                chat_response += f"💾 对话状态：\n"
                chat_response += f"   • 当前聊天历史长度: {len(updated_history)} 字符\n"
                chat_response += f"   • 已保存图像数据供下轮使用: {'是' if result_b64 else '否'}\n\n"
                chat_response += f"📋 聊天历史记录：\n{updated_history[:500]}{'...' if len(updated_history) > 500 else ''}"
                
                print(f"[APIImageEdit] 聊天模式：{operation_type}成功")
                print(f"[APIImageEdit] 自动保存聊天历史，当前长度: {len(updated_history)}")
                print(f"[APIImageEdit] 已保存图像base64数据供下一轮使用")
                print(f"[APIImageEdit] 📋 聊天总结:")
                print(chat_response)
                
                return (result_tensor, chat_response)
        
        # 失败时的处理
        print(f"[APIImageEdit] 聊天模式：{operation_type}失败")
        if images:
            return (images[0], f"聊天模式{operation_type}失败")
        else:
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, f"聊天模式{operation_type}失败")
    
    def _handle_edit_history_mode(self, api_provider, api_key, model, prompt, edit_history, images, actual_seed=-1):
        """处理编辑历史模式"""
        print(f"[APIImageEdit] 编辑历史模式启动，历史记录长度: {len(edit_history) if edit_history else 0}")
        
        # 构建包含编辑历史的完整提示词
        if edit_history.strip():
            full_prompt = f"Previous editing history:\n{edit_history}\n\nCurrent edit request: {prompt}"
            print(f"[APIImageEdit] 使用编辑历史构建上下文")
        else:
            full_prompt = prompt
            print(f"[APIImageEdit] 首次编辑，无历史记录")
        
        # 编辑历史模式需要图像输入
        if not images:
            print(f"[APIImageEdit] 编辑历史模式需要图像输入")
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_image, "编辑历史模式需要图像输入")
        
        # 编辑输入图像
        pil_img = self.tensor_to_pil(images[0])
        image_b64 = self.image_to_base64(pil_img)
        
        config = self.api_configs.get(api_provider)
        if not config:
            return (images[0], f"不支持的API提供商: {api_provider}")
            
        api_type = config.get("type", "unknown")
        
        generation_kwargs = {"seed": actual_seed} if actual_seed != -1 else {}
        try:
            if api_type == "openai_compatible":
                result_b64 = self.call_openai_compatible_api(api_provider, image_b64, full_prompt, model, api_key, **generation_kwargs)
            elif api_type == "dashscope":
                result_b64 = self.call_dashscope_api(image_b64, full_prompt, model, api_key, **generation_kwargs)
            elif api_type == "gemini":
                result_b64 = self.call_gemini_api(image_b64, full_prompt, model, api_key, **generation_kwargs)
            else:
                result_b64 = None
        except Exception as e:
            print(f"[APIImageEdit] 编辑历史模式API调用异常: {e}")
            result_b64 = None
        
        if result_b64:
            result_image = self.base64_to_image(result_b64)
            if result_image:
                result_tensor = self.pil_to_tensor(result_image)
                
                # 构建新的编辑历史记录
                import time
                timestamp = time.strftime("%H:%M:%S")
                new_edit_entry = f"[{timestamp}] {prompt}"
                
                updated_history = f"{edit_history}\n{new_edit_entry}".strip() if edit_history else new_edit_entry
                
                print(f"[APIImageEdit] 编辑历史模式：编辑成功")
                return (result_tensor, updated_history)
        
        # 失败时返回原图像
        print(f"[APIImageEdit] 编辑历史模式：编辑失败")
        return (images[0], "编辑失败")

NODE_CLASS_MAPPINGS = {
    "APIImageEditNode": APIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIImageEditNode": "API Image Edit (Enhanced)"
}