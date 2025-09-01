# ComfyUI API Image Edit

**中文说明**

强大的ComfyUI API图像编辑自定义节点，支持多个API提供商。通过ModelScope、OpenRouter、OpenAI、Google Gemini和PixelWords等API实现文本生图、单图编辑、多图合成和**多轮对话编辑**功能，具备动态模型加载、智能上下文管理和安全密钥管理能力。

**English Description**

A powerful ComfyUI custom node for API-based image editing with multiple provider support. Features text-to-image generation, single image editing, multi-image composition, and **multi-turn conversation editing** through various APIs with dynamic model loading, intelligent context management, and secure key management.

## 🚀 Features

### 多API提供商支持
- **ModelScope** - 千问图像编辑模型
- **OpenRouter** - 多种视觉模型访问
- **OpenAI** - GPT-4 Vision 和 DALL-E 模型  
- **Google Gemini** - Gemini 2.0/2.5 Flash 图像生成
- **PixelWords** - API服务商，支持Gemini 2.5 Flash Image和Gemini 2.5 Flash Image HD

### 灵活的图像输入
- **文本生图** - 仅通过文字描述创建图像
- **单图编辑** - 用自然语言编辑现有图像
- **多图合成** - 合并最多4张图像进行复杂操作
- **可选图像端口** - 所有图像输入（image1-4）都是可选的

### 高级功能
- **动态模型加载** - 自动获取每个提供商的可用模型
- **API密钥管理** - 安全的本地存储和提供商切换
- **多语言支持** - 支持中文和英文提示词
- **多轮对话模式** - 基于官方Gemini API的连续图像编辑
- **智能上下文** - 自动保持对话历史和图像状态
- **错误处理** - 强大的回退机制和清晰的错误消息

### Multi-Provider Support
- **ModelScope** - Qwen image editing models
- **OpenRouter** - Access to multiple vision models  
- **OpenAI** - GPT-4 Vision and DALL-E models
- **Google Gemini** - Gemini 2.0/2.5 Flash image generation
- **PixelWords** - API service provider supporting Gemini 2.5 Flash Image and Gemini 2.5 Flash Image HD

### Flexible Image Input
- **Text-to-Image Generation** - Create images from text descriptions only
- **Single Image Editing** - Edit existing images with natural language
- **Multi-Image Composition** - Combine up to 4 images for complex operations
- **Optional Image Ports** - All image inputs (image1-4) are optional

### Advanced Features
- **Dynamic Model Loading** - Automatically fetch available models for each provider
- **API Key Management** - Secure local storage with provider switching
- **Multi-Language Support** - Supports Chinese and English prompts
- **Multi-Turn Conversation** - Continuous image editing based on official Gemini API
- **Intelligent Context** - Automatically maintains conversation history and image state
- **Error Handling** - Robust fallback mechanisms and clear error messages

## 📦 安装方法

### 方式1: ComfyUI Manager (推荐)
1. 通过ComfyUI Manager安装
2. 搜索 "ComfyUI API Image Edit"
3. 点击安装

### 方式2: 手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aiaiaikkk/comfyui-api-image-edit.git
cd comfyui-api-image-edit
pip install -r requirements.txt
```

### 方式3: PyPI安装
```bash
pip install comfyui-api-image-edit
```

## 📦 Installation

### Option 1: ComfyUI Manager (Recommended)
1. Install through ComfyUI Manager
2. Search for "ComfyUI API Image Edit"
3. Click Install

### Option 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aiaiaikkk/comfyui-api-image-edit.git
cd comfyui-api-image-edit
pip install -r requirements.txt
```

### Option 3: PyPI Installation
```bash
pip install comfyui-api-image-edit
```

## 🔧 Dependencies
- `requests>=2.25.1` - HTTP API calls
- `Pillow>=8.0.0` - Image processing
- `numpy>=1.19.0` - Array operations  
- `torch>=1.9.0` - ComfyUI tensor operations
- `google-genai>=1.32.0` - Google Gemini API support

## 🎯 使用方法

### 基本设置
1. 将"API Image Edit"节点添加到ComfyUI工作流
2. 选择你喜欢的API提供商
3. 输入你的API密钥
4. 从下拉菜单中选择模型
5. 编写你的提示词

### 生成模式

#### 文本生图
- 不要连接任何图像到输入端口
- 在提示词中写下你的描述
- 节点将根据文本生成新图像

#### 单图编辑
- 将图像连接到 `image1` 端口
- 在提示词中描述你想要的更改
- 示例："将背景更改为日落场景"

#### 多图合成
- 将2-4张图像连接到 `image1`, `image2`, `image3`, `image4` 端口
- 在提示词中描述合成方式
- 示例："将这些图像合并为拼贴画" 或 "用image2中的脸替换image1中的脸"

#### 多轮对话编辑 🆕
- **连续编辑模式**：保持图像编辑的连续性，无需重新上传图像
- **使用方法**：
  1. 第一轮：连接图像到 `image1`，输入提示词如"让这个人笑"
  2. 第二轮：**断开** `image1` 连接，启用 `use_last_image`，输入"戴上红帽子"
  3. 第三轮：继续输入"改变背景为花园"，系统会基于前面结果继续编辑
- **控制参数**：
  - `conversation_mode`: 启用多轮对话模式（默认开启）
  - `use_last_image`: 使用对话历史中的最后一张图像
  - `reset_conversation`: 重置对话开始新的编辑序列

#### 智能工作流程
```
输入图片 → "笑" → 生成笑脸
    ↓
断开连接 + use_last_image → "戴花" → 在笑脸基础上戴花
    ↓
继续对话 → "海滩背景" → 在戴花笑脸基础上换背景
```

## 🎯 Usage

### Basic Setup
1. Add the "API Image Edit" node to your ComfyUI workflow
2. Select your preferred API provider
3. Enter your API key
4. Choose a model from the dropdown
5. Write your prompt

### Generation Modes

#### Text-to-Image
- Don't connect any images to the input ports
- Write your description in the prompt
- The node will generate new images based on your text

#### Single Image Editing
- Connect an image to `image1` port
- Describe the changes you want in the prompt
- Example: "Change the background to a sunset scene"

#### Multi-Image Composition
- Connect 2-4 images to `image1`, `image2`, `image3`, `image4` ports
- Describe the composition in the prompt
- Example: "Combine these images into a collage" or "Replace the face in image1 with the face from image2"

#### Multi-Turn Conversation Editing 🆕
- **Continuous Editing Mode**: Maintains continuity in image editing without re-uploading images
- **How to Use**:
  1. First Turn: Connect image to `image1`, input prompt like "make this person smile"
  2. Second Turn: **Disconnect** `image1`, enable `use_last_image`, input "add a red hat"
  3. Third Turn: Continue with "change background to garden", system will edit based on previous results
- **Control Parameters**:
  - `conversation_mode`: Enable multi-turn conversation mode (default: enabled)
  - `use_last_image`: Use the last image from conversation history
  - `reset_conversation`: Reset conversation to start a new editing sequence

#### Smart Workflow
```
Input Image → "smile" → Generate Smiling Face
    ↓
Disconnect + use_last_image → "add flower" → Add flower to smiling face
    ↓
Continue conversation → "beach background" → Change background while keeping flower and smile
```

### API提供商配置

#### ModelScope
- 从 [ModelScope](https://modelscope.cn) 获取API密钥
- 支持千问图像编辑模型
- 最适合中文提示词

#### OpenRouter
- 从 [OpenRouter](https://openrouter.ai/keys) 获取API密钥
- 通过单个API访问多个提供商
- 支持各种视觉模型

#### OpenAI
- 从 [OpenAI](https://platform.openai.com) 获取API密钥
- 支持GPT-4 Vision和DALL-E模型
- 高质量图像生成

#### Google Gemini
- 从 [Google AI Studio](https://makersuite.google.com) 获取API密钥
- 支持最新的Gemini 2.0 Flash模型
- 高级多模态能力

#### PixelWords
- API服务商平台
- 支持Gemini 2.5 Flash Image和Gemini 2.5 Flash Image HD
- 通过第三方API访问Google图像生成模型

### API Provider Configuration

#### ModelScope
- Get API key from [ModelScope](https://modelscope.cn)
- Supports Qwen image editing models
- Best for Chinese language prompts

#### OpenRouter
- Get API key from [OpenRouter](https://openrouter.ai/keys)
- Access to multiple providers through one API
- Supports various vision models

#### OpenAI
- Get API key from [OpenAI](https://platform.openai.com)
- Supports GPT-4 Vision and DALL-E models
- High quality image generation

#### Google Gemini
- Get API key from [Google AI Studio](https://makersuite.google.com)
- Supports latest Gemini 2.0 Flash models
- Advanced multimodal capabilities

#### PixelWords
- API service platform
- Supports Gemini 2.5 Flash Image and Gemini 2.5 Flash Image HD
- Third-party API access to Google's image generation models

## 🛠️ Development

### Building from Source
```bash
git clone https://github.com/aiaiaikkk/comfyui-api-image-edit.git
cd comfyui-api-image-edit
pip install -e .
```

### Running Tests
```bash
pip install -e .[dev]
pytest
```

## 📝 Configuration

### Node Parameters

#### 必需参数 / Required Parameters
- **API Provider** - Select from available providers / 选择API提供商
- **API Key** - Your API key (stored locally and securely) / API密钥（本地安全存储）
- **Model** - Auto-populated based on provider selection / 基于提供商自动填充的模型选择
- **Prompt** - Describe your desired image or edits / 描述您想要的图像或编辑

#### 图像输入 / Image Inputs  
- **Image1-4** - Optional image inputs for editing/composition / 用于编辑/合成的可选图像输入

#### 多轮对话控制 / Multi-Turn Conversation Controls 🆕
- **conversation_mode** - Enable/disable conversation mode / 启用/禁用对话模式 (默认: True)
- **use_last_image** - Use last image from history / 使用历史中的最后一张图像 (默认: False)  
- **reset_conversation** - Reset current conversation / 重置当前对话 (默认: False)

#### 高级设置 / Advanced Settings
- **Strength** - Editing intensity (0.0-2.0) / 编辑强度
- **Guidance Scale** - Generation guidance (1.0-20.0) / 生成引导系数
- **Steps** - Sampling steps (1-100) / 采样步数
- **Seed** - Random seed for reproducibility / 可重现性的随机种子
- **Negative Prompt** - What to avoid in generation / 生成中要避免的内容
- **Watermark** - Add watermark to generated images / 为生成图像添加水印
- **Refresh Models** - Update model list from API / 从API更新模型列表

## 🔒 Privacy & Security
- API keys are stored locally in browser localStorage
- No data is sent to third parties except chosen API providers
- All processing happens through official API endpoints
- Supports proxy configurations for enhanced privacy

## 🐛 Troubleshooting

### Common Issues
1. **"API key is required"** - Enter a valid API key for selected provider
2. **"Model not available"** - Click refresh models or switch providers
3. **"Import Error"** - Install missing dependencies with `pip install -r requirements.txt`
4. **Network errors** - Check internet connection and API key validity

### Error Messages
- Clear error messages guide you to solutions
- Automatic fallback to REST APIs when SDK unavailable
- Detailed logging for debugging

### 多轮对话常见问题 / Multi-Turn Conversation FAQ 🆕

**Q: 如何实现连续图像编辑？ / How to achieve continuous image editing?**
A: 第一轮连接图像，之后断开连接并启用 `use_last_image`，系统会自动使用上一轮结果。
   First turn connect image, then disconnect and enable `use_last_image`, system will use previous results automatically.

**Q: 什么时候需要重置对话？ / When should I reset conversation?**
A: 开始全新编辑任务时启用 `reset_conversation`，清除历史记录。
   Enable `reset_conversation` when starting a completely new editing task to clear history.

**Q: 支持哪些API提供商的多轮对话？ / Which providers support multi-turn conversation?**
A: 目前完整支持 Google Gemini，其他提供商的支持正在开发中。
   Currently full support for Google Gemini, support for other providers is under development.

**Q: 对话历史会占用多少内存？ / How much memory does conversation history use?**
A: 自动限制最多20条消息和10张图像，避免内存溢出。
   Automatically limits to 20 messages and 10 images maximum to prevent memory overflow.

## 📄 License
MIT License - see LICENSE file for details

## 🤝 Contributing
Contributions welcome! Please feel free to submit pull requests or open issues.

## 🔗 Links
- [GitHub Repository](https://github.com/aiaiaikkk/comfyui-api-image-edit)
- [Issue Tracker](https://github.com/aiaiaikkk/comfyui-api-image-edit/issues)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## ⭐ Support
If you find this project helpful, please give it a star on GitHub!