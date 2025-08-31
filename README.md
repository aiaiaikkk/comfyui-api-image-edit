# ComfyUI API Image Edit

A powerful ComfyUI custom node for API-based image editing with multiple provider support.

## 🚀 Features

### Multi-Provider Support
- **ModelScope** - Qwen image editing models
- **OpenRouter** - Access to multiple vision models  
- **OpenAI** - GPT-4 Vision and DALL-E models
- **Google Gemini** - Gemini 2.0 Flash image generation
- **PixelWords** - Advanced image generation models

### Flexible Image Input
- **Text-to-Image Generation** - Create images from text descriptions only
- **Single Image Editing** - Edit existing images with natural language
- **Multi-Image Composition** - Combine up to 4 images for complex operations
- **Optional Image Ports** - All image inputs (image1-4) are optional

### Advanced Features
- **Dynamic Model Loading** - Automatically fetch available models for each provider
- **API Key Management** - Secure local storage with provider switching
- **Multi-Language Support** - Supports Chinese and English prompts
- **Error Handling** - Robust fallback mechanisms and clear error messages

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

### API Provider Configuration

#### ModelScope
- Get API key from [ModelScope](https://modelscope.cn)
- Supports Qwen image editing models
- Best for Chinese language prompts

#### OpenRouter
- Get API key from [OpenRouter](https://openrouter.ai)
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
- Advanced image generation platform
- Multiple specialized models available
- High-quality artistic outputs

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
- **API Provider** - Select from available providers
- **API Key** - Your API key (stored locally and securely)
- **Model** - Auto-populated based on provider selection
- **Prompt** - Describe your desired image or edits
- **Image1-4** - Optional image inputs for editing/composition
- **Refresh Models** - Update model list from API
- **Advanced Settings** - Strength, guidance scale, steps, etc.

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