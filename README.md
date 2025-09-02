# ComfyUI API Image Edit

一个强大的ComfyUI自定义节点，支持通过多种远程API进行图片编辑功能。

## 🌟 功能特性

- 🔌 **多API支持**: 支持阿里云千问和OpenRouter API
- 🤖 **自动模型发现**: 自动获取并更新支持的模型列表
- 🎨 **智能图片编辑**: 基于文本提示的AI图片编辑
- 🎭 **蒙版支持**: 支持使用蒙版进行局部编辑
- ⚡ **高效缓存**: 智能缓存模型列表，减少API调用
- 🛡️ **错误处理**: 完善的错误处理和回退机制

## 📋 支持的API提供商

### 1. 阿里云千问 (DashScope)
- **模型**: qwen-image-edit (专用图片编辑模型), qwen-vl-max, qwen-vl-plus
- **认证**: API Key (需要在环境变量DASHSCOPE_API_KEY中设置或直接输入)
- **功能**: 专业的AI图片编辑，支持IP创作、文字替换、风格迁移、人物换装等

### 2. OpenRouter
- **模型**: Google Gemini 2.5 Flash Image Preview, Gemini 2.0 Flash, LLaMA Vision 等
- **认证**: API Key
- **功能**: 图像理解与描述 (注意：大多数Vision模型主要用于分析而非生成图像)

## 🚀 安装

1. 将此文件夹复制到 ComfyUI 的 `custom_nodes` 目录
2. 重启 ComfyUI
3. 在节点列表中找到 "API/Image Edit" 分类

## 📖 使用方法

### 基本使用

1. **添加节点**: 在ComfyUI中搜索 "API Image Edit"
2. **选择API提供商**: 从下拉菜单选择 "阿里云千问" 或 "OpenRouter"
3. **输入API Key**: 在API Key字段输入您的密钥
4. **选择模型**: 输入模型名称或点击刷新获取可用模型列表
5. **输入提示**: 在prompt字段描述您想要的编辑效果
6. **连接图片**: 将图片输入连接到image接口

### 高级功能

- **蒙版编辑**: 连接MASK输入进行局部编辑
- **参数调节**: 
  - `strength`: 编辑强度 (0.0-2.0)
  - `guidance_scale`: 引导强度 (1.0-20.0) 
  - `steps`: 处理步数 (1-100)
  - `seed`: 随机种子 (-1为随机)
  - `watermark`: 是否添加水印 (阿里云千问专用)
  - `negative_prompt`: 负面提示词 (描述不想要的内容)

## 🔑 获取API密钥

### 阿里云千问
1. 访问 [阿里云模型服务平台](https://dashscope.console.aliyun.com/)
2. 注册并创建API Key
3. 确保账户有足够余额

### OpenRouter
1. 访问 [OpenRouter](https://openrouter.ai/)
2. 注册账户并获取API Key
3. 查看支持的模型列表

## ⚙️ 配置说明

### 节点输入

**必需参数:**
- `image`: 输入图片
- `api_provider`: API提供商选择
- `api_key`: API密钥
- `model`: 模型名称
- `prompt`: 编辑提示文本

**可选参数:**
- `mask`: 编辑蒙版
- `refresh_models`: 刷新模型列表
- `strength`: 编辑强度
- `guidance_scale`: 引导强度  
- `steps`: 处理步数
- `seed`: 随机种子
- `watermark`: 添加水印 (阿里云千问)
- `negative_prompt`: 负面提示词

## 🔧 故障排除

### 常见问题

1. **API调用失败**
   - 检查API Key是否正确
   - 检查网络连接
   - 确认账户余额充足

2. **模型列表为空**
   - 点击刷新模型按钮
   - 检查API Key权限
   - 使用默认模型名称

3. **图片处理失败**
   - 检查图片格式是否支持
   - 确认prompt是否合理
   - 尝试调整参数

### 调试模式

节点会在控制台输出详细的日志信息，包括:
- API调用状态
- 可用模型列表
- 错误信息和堆栈跟踪

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 支持

如果您遇到问题或有建议，请在GitHub上创建Issue。