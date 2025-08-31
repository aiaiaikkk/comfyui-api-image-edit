/**
 * ComfyUI API Image Edit - 完全按照参考项目设计的前端UI
 * 复刻super-prompt-canvas的远程API界面设计
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// API提供商配置 - 包含5个主要提供商
const API_PROVIDERS = {
    'modelscope': {
        name: 'ModelScope',
        models: ['Qwen/Qwen-Image-Edit', 'MusePublic/Qwen-Image-Edit', 'qwen-image-edit', 'Qwen/Qwen-Image'],
        baseUrl: 'https://api-inference.modelscope.cn/v1',
        keyPlaceholder: '输入ModelScope API Token...',
        supportsDynamic: true
    },
    'openrouter': {
        name: 'OpenRouter',
        models: ['google/gemini-2.5-flash-image-preview:free', 'google/gemini-2.0-flash-exp:free'],
        baseUrl: 'https://openrouter.ai/api/v1',
        keyPlaceholder: '输入API密钥/访问令牌...',
        supportsDynamic: true
    },
    'openai': {
        name: 'OpenAI',
        models: ['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'],
        baseUrl: 'https://api.openai.com/v1',
        keyPlaceholder: '输入API密钥/访问令牌...',
        supportsDynamic: true
    },
    'gemini': {
        name: 'Google Gemini',
        models: ['gemini-2.0-flash-preview-image-generation', 'gemini-2.5-flash-image-preview'],
        baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
        keyPlaceholder: '输入Google API密钥...',
        supportsDynamic: true
    },
    'pixelwords': {
        name: 'PixelWords',
        models: ['gemini-2.5-flash-image-preview', 'gpt-4o-image', 'ideogram', 'flux-kontext-max'],
        baseUrl: 'https://api.sydney-ai.com/v1',
        keyPlaceholder: '输入PixelWords API密钥...',
        supportsDynamic: true
    }
};

// API密钥管理器 - 完全复制参考项目的逻辑
class APIKeyManager {
    constructor() {
        this.STORAGE_KEY = "api_image_edit_keys";
        this.PROVIDER_KEY = "api_image_edit_provider";
    }

    saveKey(provider, key) {
        try {
            const keys = this.getAllKeys();
            keys[provider] = key;
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(keys));
        } catch (e) {
            console.error("[APIImageEdit] 保存密钥失败:", e);
        }
    }

    getKey(provider) {
        try {
            const keys = this.getAllKeys();
            return keys[provider] || "";
        } catch (e) {
            return "";
        }
    }

    saveProvider(provider) {
        try {
            localStorage.setItem(this.PROVIDER_KEY, provider);
        } catch (e) {
            console.error("[APIImageEdit] 保存provider失败:", e);
        }
    }

    getSavedProvider() {
        try {
            return localStorage.getItem(this.PROVIDER_KEY) || "modelscope";
        } catch (e) {
            return "modelscope";
        }
    }

    getAllKeys() {
        try {
            const stored = localStorage.getItem(this.STORAGE_KEY);
            return stored ? JSON.parse(stored) : {};
        } catch (e) {
            return {};
        }
    }
}

// 模型获取器 - 动态获取API模型列表
class ModelFetcher {
    constructor() {
        this.cache = new Map();
        this.fetchPromises = new Map();
    }

    async fetchModels(provider, apiKey) {
        const cacheKey = `${provider}_${apiKey.substring(0, 10)}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        if (this.fetchPromises.has(cacheKey)) {
            return await this.fetchPromises.get(cacheKey);
        }

        const config = API_PROVIDERS[provider];
        if (!config || !config.supportsDynamic) {
            return config?.models || [];
        }

        const fetchPromise = this._doFetchModels(provider, apiKey, config);
        this.fetchPromises.set(cacheKey, fetchPromise);

        try {
            const models = await fetchPromise;
            this.cache.set(cacheKey, models);
            return models;
        } catch (error) {
            console.warn(`[APIImageEdit] 获取${provider}模型失败:`, error);
            return config.models || [];
        } finally {
            this.fetchPromises.delete(cacheKey);
        }
    }

    async _doFetchModels(provider, apiKey, config) {
        if (provider === 'openrouter') {
            return await this._fetchOpenRouterModels(apiKey);
        } else if (provider === 'gemini') {
            return await this._fetchGeminiModels(apiKey);
        } else if (provider === 'claude') {
            return await this._fetchClaudeModels(apiKey);
        } else if (provider === 'modelscope') {
            // ModelScope doesn't have a models API endpoint, return default models
            console.log("[APIImageEdit] ModelScope使用预定义模型列表");
            return config.models || [];
        } else if (provider === 'pixelwords') {
            return await this._fetchPixelWordsModels(apiKey);
        } else {
            return await this._fetchOpenAICompatibleModels(config.baseUrl, apiKey);
        }
    }

    async _fetchOpenRouterModels(apiKey) {
        const response = await fetch('https://openrouter.ai/api/v1/models', {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://comfyui.com',
                'X-Title': 'ComfyUI API Image Edit'
            }
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        return data.data?.filter(model => {
            const id = model.id.toLowerCase();
            return id.includes('vision') || id.includes('gemini') || 
                   id.includes('gpt-4') || id.includes('claude') || 
                   id.includes('image') || id.includes('multimodal');
        }).map(model => model.id) || [];
    }

    async _fetchGeminiModels(apiKey) {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        return data.models
            ?.filter(model => model.supportedGenerationMethods?.includes('generateContent'))
            ?.map(model => model.name.replace('models/', '')) || [];
    }

    async _fetchClaudeModels(apiKey) {
        const response = await fetch('https://api.anthropic.com/v1/models', {
            headers: {
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        return data.data?.map(model => model.id) || [];
    }

    async _fetchPixelWordsModels(apiKey) {
        const response = await fetch('https://api.sydney-ai.com/v1/models', {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        // 过滤图像相关模型 - 更精确的筛选
        const imageModels = data.data?.filter(model => {
            const id = model.id.toLowerCase();
            return id.includes('image') || id.includes('vision') || 
                   id.includes('gemini-2.5-flash-image') || id.includes('gemini-2.0-flash-preview-image') ||
                   id.includes('gpt-4o-image') || id.includes('gpt-4-dalle') || id.includes('dall-e') || 
                   id.includes('stable-diffusion') || id.includes('flux') || id.includes('ideogram') || 
                   id.includes('mj-') || id.includes('midjourney') ||
                   id.includes('grok-3-imagegen') || id.includes('seededit') || id.includes('glm-4v') ||
                   id.includes('llava') || id.includes('playground') || id.includes('kling_image') ||
                   id.includes('avatar') || id.includes('sd3') || id.includes('ssd-') ||
                   id.includes('chat-seedream') || id.includes('api-images');
        }).map(model => model.id) || [];
        
        console.log(`[APIImageEdit] PixelWords获取到 ${imageModels.length} 个图像模型:`, imageModels.slice(0, 5));
        return imageModels;
    }

    async _fetchOpenAICompatibleModels(baseUrl, apiKey) {
        const response = await fetch(`${baseUrl}/models`, {
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        return data.data?.map(model => model.id) || [];
    }
}

// 全局实例
const keyManager = new APIKeyManager();
const modelFetcher = new ModelFetcher();

// 主要的节点扩展
app.registerExtension({
    name: "ComfyUI.APIImageEditUI",
    
    async nodeCreated(node) {
        if (node.type === "APIImageEditNode" || node.comfyClass === "APIImageEditNode") {
            setTimeout(() => {
                this.enhanceAPIImageEditNode(node);
            }, 100);
        }
    },

    enhanceAPIImageEditNode(node) {
        console.log("[APIImageEdit] 增强节点UI，匹配参考项目设计");
        
        // 找到widgets
        const providerWidget = node.widgets?.find(w => w.name === "api_provider");
        const keyWidget = node.widgets?.find(w => w.name === "api_key");
        const modelWidget = node.widgets?.find(w => w.name === "model");
        const refreshWidget = node.widgets?.find(w => w.name === "refresh_models");

        if (!providerWidget || !keyWidget || !modelWidget) {
            console.warn("[APIImageEdit] 找不到必要的widgets");
            return;
        }

        // 增强API Provider widget
        this.enhanceProviderWidget(node, providerWidget, keyWidget, modelWidget);
        
        // 增强API Key widget
        this.enhanceKeyWidget(node, keyWidget, providerWidget);
        
        // 增强Model widget
        this.enhanceModelWidget(node, modelWidget, providerWidget, keyWidget);
        
        // 增强Refresh widget
        if (refreshWidget) {
            this.enhanceRefreshWidget(node, refreshWidget, providerWidget, keyWidget, modelWidget);
        }

        // 初始化节点
        this.initializeNode(node, providerWidget, keyWidget, modelWidget);
    },

    enhanceProviderWidget(node, providerWidget, keyWidget, modelWidget) {
        // 恢复保存的提供商
        const savedProvider = keyManager.getSavedProvider();
        if (savedProvider && API_PROVIDERS[savedProvider]) {
            // 找到对应的显示名称
            const providerName = API_PROVIDERS[savedProvider].name;
            if (providerWidget.options.values.includes(providerName)) {
                providerWidget.value = providerName;
            }
        }

        // 监听提供商变化
        const originalCallback = providerWidget.callback;
        providerWidget.callback = (value) => {
            console.log(`[APIImageEdit] 🔄 提供商切换: ${providerWidget._previousValue || '未知'} -> ${value}`);
            
            if (originalCallback) {
                originalCallback.call(providerWidget, value);
            }
            
            // 保存当前提供商的密钥（如果有的话）
            const currentProviderKey = this.getProviderKeyFromName(providerWidget._previousValue || value);
            const newProviderKey = this.getProviderKeyFromName(value);
            
            if (keyWidget._realValue && currentProviderKey !== newProviderKey && providerWidget._previousValue) {
                console.log(`[APIImageEdit] 💾 保存 ${currentProviderKey} 的API密钥: ${keyWidget._realValue.substring(0, 10)}...`);
                keyManager.saveKey(currentProviderKey, keyWidget._realValue);
            }
            
            // 获取新provider key
            const providerKey = this.getProviderKeyFromName(value);
            keyManager.saveProvider(providerKey);
            
            // 恢复新提供商的保存密钥
            const savedKey = keyManager.getKey(providerKey);
            if (savedKey) {
                console.log(`[APIImageEdit] 恢复 ${providerKey} 的API密钥`);
                keyWidget.value = savedKey;
                keyWidget._realValue = savedKey;
                keyWidget._isHidden = false;
                if (keyWidget.callback) {
                    keyWidget.callback(savedKey);
                }
                // 延时隐藏
                setTimeout(() => {
                    this.hideAPIKey(keyWidget);
                }, 1000);
            } else {
                // 没有保存的密钥，清空输入框
                keyWidget.value = "";
                keyWidget._realValue = "";
                keyWidget._isHidden = false;
            }
            
            // 记住当前选择的提供商，用于下次切换时保存密钥
            providerWidget._previousValue = value;
            
            // 更新API Key占位符
            const config = API_PROVIDERS[providerKey];
            // 配置占位符在ComfyUI中通过其他方式处理
            
            // 更新模型列表 - 强制更新
            this.updateModelList(node, providerKey, keyWidget.value || keyWidget._realValue, modelWidget, true);
        };
    },

    enhanceKeyWidget(node, keyWidget, providerWidget) {
        // 存储真实的API key值
        keyWidget._realValue = keyWidget.value || '';
        keyWidget._isHidden = false;

        // 保存原始的value属性
        keyWidget._originalValue = keyWidget.value;
        
        // 简化方案：重写serialize方法
        const originalSerialize = keyWidget.serialize;
        keyWidget.serialize = function() {
            console.log(`[APIImageEdit] 🔍 SERIALIZE被调用! value: "${this.value}", _realValue: "${this._realValue}", _isHidden: ${this._isHidden}`);
            const result = originalSerialize ? originalSerialize.call(this) : this.value;
            // 如果当前显示的是●掩码，返回真实值
            if (this.value && this.value.match(/^●+$/) && this._realValue) {
                console.log('[APIImageEdit] ✅ 序列化时使用真实API key');
                return this._realValue;
            }
            console.log('[APIImageEdit] ⚠️ 序列化时使用显示值');
            return result;
        };
        
        // 额外测试：添加serializeValue方法（某些版本可能使用这个）
        if (!keyWidget.serializeValue) {
            keyWidget.serializeValue = keyWidget.serialize;
        }

        // 监听密钥变化
        const originalCallback = keyWidget.callback;
        keyWidget.callback = (value) => {
            // 如果输入的是●符号，说明是隐藏状态，使用真实值
            if (value && value.match(/^●+$/)) {
                return; // 隐藏状态不处理
            }
            
            // 正常输入的API key
            if (value && value.trim()) {
                keyWidget._realValue = value.trim();
                keyWidget._isHidden = false;
                
                if (originalCallback) {
                    originalCallback.call(keyWidget, value);
                }
                
                const providerName = providerWidget.value;
                const providerKey = this.getProviderKeyFromName(providerName);
                keyManager.saveKey(providerKey, value.trim());
                
                // 更新模型列表
                const modelWidget = node.widgets?.find(w => w.name === "model");
                if (modelWidget) {
                    console.log(`[APIImageEdit] API key变化，刷新模型列表: ${providerKey}`);
                    this.updateModelList(node, providerKey, value.trim(), modelWidget, true);
                }
                
                // 延时隐藏API key
                setTimeout(() => {
                    this.hideAPIKey(keyWidget);
                }, 1000);
            } else if (originalCallback) {
                originalCallback.call(keyWidget, value);
            }
        };

        // 增强API Key功能
        setTimeout(() => {
            const providerName = providerWidget.value;
            const providerKey = this.getProviderKeyFromName(providerName);
            
            // 如果已经有保存的key，显示为隐藏状态
            if (keyWidget.value && keyWidget.value.length > 0) {
                keyWidget._realValue = keyWidget.value;
                setTimeout(() => {
                    this.hideAPIKey(keyWidget);
                }, 100);
            }
        }, 200);
    },

    hideAPIKey(keyWidget) {
        if (keyWidget._realValue && keyWidget._realValue.length > 0 && !keyWidget._isHidden) {
            const hiddenValue = '●'.repeat(Math.min(keyWidget._realValue.length, 20));
            // 直接设置value而不是_displayValue
            keyWidget.value = hiddenValue;
            keyWidget._isHidden = true;
            console.log(`[APIImageEdit] API key已隐藏: ${hiddenValue}`);
            
            // 手动更新DOM显示
            if (keyWidget.inputEl) {
                keyWidget.inputEl.value = hiddenValue;
            }
        }
    },

    // 获取真实的API Key值
    getRealAPIKey(keyWidget) {
        return keyWidget._realValue || keyWidget.value;
    },

    enhanceModelWidget(node, modelWidget, providerWidget, keyWidget) {
        // 为模型widget添加加载状态指示
        modelWidget._originalStyle = null;
    },

    async updateModelList(node, providerKey, apiKey, modelWidget, forceUpdate = false) {
        if (!modelWidget) return;

        const config = API_PROVIDERS[providerKey];
        if (!config) return;

        try {
            console.log(`[APIImageEdit] 更新${config.name}模型列表，provider: ${providerKey}`);
            
            // 显示加载状态
            this.setModelLoadingState(modelWidget, true);

            let models;
            
            // 首先使用默认模型，确保提供商匹配
            models = config.models || [];
            
            // 如果有API key且支持动态获取，尝试获取更多模型
            if (apiKey && apiKey.trim() && config.supportsDynamic && !apiKey.match(/^●+$/)) {
                try {
                    const dynamicModels = await modelFetcher.fetchModels(providerKey, apiKey.trim());
                    if (dynamicModels && dynamicModels.length > 0) {
                        models = dynamicModels;
                    }
                } catch (error) {
                    console.warn(`[APIImageEdit] 动态获取模型失败，使用默认模型:`, error);
                }
            }

            if (models.length > 0) {
                // 更新COMBO widget的选项
                modelWidget.options.values = models;
                
                // 选择第一个模型
                modelWidget.value = models[0];
                
                console.log(`[APIImageEdit] 成功更新${config.name}模型列表:`, models);
            }

        } catch (error) {
            console.error(`[APIImageEdit] 更新模型列表失败:`, error);
            // 使用默认模型作为最后的fallback
            if (config.models) {
                modelWidget.options.values = config.models;
                modelWidget.value = config.models[0];
            }
        } finally {
            this.setModelLoadingState(modelWidget, false);
        }
    },

    setModelLoadingState(modelWidget, loading) {
        if (!modelWidget.domWidget) return;

        const selectEl = modelWidget.domWidget.querySelector('select');
        if (!selectEl) return;

        if (loading) {
            if (!modelWidget._originalStyle) {
                modelWidget._originalStyle = selectEl.style.background;
            }
            selectEl.style.background = '#2a4a6b';
            selectEl.disabled = true;
            
            // 添加加载选项
            const loadingOption = document.createElement('option');
            loadingOption.value = '';
            loadingOption.textContent = '正在获取模型列表...';
            selectEl.innerHTML = '';
            selectEl.appendChild(loadingOption);
        } else {
            selectEl.style.background = modelWidget._originalStyle || '';
            selectEl.disabled = false;
        }
    },

    async initializeNode(node, providerWidget, keyWidget, modelWidget) {
        // 恢复保存的设置
        const savedProvider = keyManager.getSavedProvider();
        if (savedProvider && API_PROVIDERS[savedProvider]) {
            // 设置provider
            const providerName = API_PROVIDERS[savedProvider].name;
            if (providerWidget.options.values.includes(providerName)) {
                providerWidget.value = providerName;
                // 设置_previousValue以便切换时能正确保存密钥
                providerWidget._previousValue = providerName;
            }
            
            // 恢复API key
            const savedKey = keyManager.getKey(savedProvider);
            if (savedKey) {
                console.log(`[APIImageEdit] 初始化时恢复 ${savedProvider} 的API密钥`);
                keyWidget.value = savedKey;
                keyWidget._realValue = savedKey;
                keyWidget._isHidden = false;
            } else {
                // 清空密钥字段
                keyWidget.value = "";
                keyWidget._realValue = "";
                keyWidget._isHidden = false;
            }
            
            // 设置占位符
            const config = API_PROVIDERS[savedProvider];
            // 配置占位符在ComfyUI中通过其他方式处理
            
            // 强制更新模型列表以匹配当前提供商
            console.log(`[APIImageEdit] 初始化节点，provider: ${savedProvider}`);
            await this.updateModelList(node, savedProvider, savedKey, modelWidget, true);
            
            // 隐藏API key
            if (savedKey) {
                setTimeout(() => {
                    this.hideAPIKey(keyWidget);
                }, 500);
            }
        } else {
            // 如果没有保存的设置，使用默认设置
            const defaultProvider = 'modelscope'; // 默认使用ModelScope
            const defaultConfig = API_PROVIDERS[defaultProvider];
            if (defaultConfig) {
                providerWidget.value = defaultConfig.name;
                providerWidget._previousValue = defaultConfig.name; // 设置_previousValue
                keyWidget.value = "";
                keyWidget._realValue = "";
                keyWidget._isHidden = false;
                await this.updateModelList(node, defaultProvider, '', modelWidget, true);
            }
        }
    },

    getProviderKeyFromName(providerName) {
        for (const [key, config] of Object.entries(API_PROVIDERS)) {
            if (config.name === providerName) {
                return key;
            }
        }
        return 'modelscope'; // 默认值
    },
    
    enhanceRefreshWidget(node, refreshWidget, providerWidget, keyWidget, modelWidget) {
        const originalCallback = refreshWidget.callback;
        refreshWidget.callback = async (value) => {
            if (originalCallback) {
                originalCallback.call(refreshWidget, value);
            }
            
            if (value === true) {
                console.log("[APIImageEdit] 手动刷新模型列表");
                const providerKey = this.getProviderKeyFromName(providerWidget.value);
                const apiKey = keyWidget._realValue || keyWidget.value;
                
                await this.updateModelList(node, providerKey, apiKey, modelWidget, true);
                
                // 重置refresh按钮
                setTimeout(() => {
                    refreshWidget.value = false;
                }, 100);
            }
        };
    }
});

console.log("[APIImageEdit] UI扩展已加载，完全匹配参考项目设计");