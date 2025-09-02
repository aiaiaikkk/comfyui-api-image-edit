"""
ComfyUI API Image Edit - Custom Node Package

一个支持多种API提供商的图片编辑自定义节点包
支持阿里云千问和OpenRouter API

Version: 1.0.0
Author: User
"""

import os
import importlib.util
import traceback

__version__ = "1.0.1"  # Fixed UnboundLocalError in single mode
__author__ = "User"
__description__ = "ComfyUI Custom Nodes for API-based Image Editing"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def get_ext_dir(subpath=None):
    """获取扩展目录路径"""
    dir_path = os.path.dirname(__file__)
    if subpath is not None:
        dir_path = os.path.join(dir_path, subpath)
    return os.path.abspath(dir_path)

def load_node_modules():
    """加载节点模块"""
    current_dir = get_ext_dir()
    
    for file in os.listdir(current_dir):
        if not file.endswith(".py") or file.startswith("__") or file == "__init__.py":
            continue
        
        if "backup" in file or "test" in file or file.endswith("_backup.py") or file.endswith("_old.py"):
            continue
        
        module_name = os.path.splitext(file)[0]
        module_path = os.path.join(current_dir, file)
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                imported_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imported_module)
                
                if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
                    NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
                    
                if hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                    NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
                    
                print(f"[ComfyUI_API_Image_Edit] ✅ 成功加载模块: {module_name}")
                
        except Exception as e:
            print(f"[ComfyUI_API_Image_Edit] ❌ 加载模块失败 {module_name}: {e}")
            traceback.print_exc()

load_node_modules()

print(f"[ComfyUI_API_Image_Edit] 📦 插件版本: {__version__}")
print(f"[ComfyUI_API_Image_Edit] 🚀 已加载 {len(NODE_CLASS_MAPPINGS)} 个节点")

if NODE_CLASS_MAPPINGS:
    print(f"[ComfyUI_API_Image_Edit] 📋 可用节点: {list(NODE_CLASS_MAPPINGS.keys())}")

# 定义Web目录
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]