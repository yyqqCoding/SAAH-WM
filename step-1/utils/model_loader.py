"""
模型加载器模块

负责加载和管理SAAH-WM所需的各种预训练模型，
包括CLIP模型和Stable Diffusion 2.1模型。

作者：SAAH-WM团队
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from typing import Tuple, Optional
import gc

from .logger_config import LoggerMixin


class ModelLoader(LoggerMixin):
    """
    模型加载器类
    
    负责加载和管理CLIP模型和Stable Diffusion模型，
    提供统一的模型访问接口。
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            device: 计算设备，如果为None则自动选择
        """
        super().__init__()
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.log_info(f"使用设备: {self.device}")
        
        # 模型存储
        self.clip_model = None
        self.clip_processor = None
        self.sd_pipeline = None
        
    def load_clip_model(self, model_name: str = "openai/clip-vit-large-patch14") -> Tuple[CLIPModel, CLIPProcessor]:
        """
        加载CLIP模型和处理器
        
        Args:
            model_name: CLIP模型名称
            
        Returns:
            (CLIP模型, CLIP处理器)
        """
        try:
            self.log_info(f"开始加载CLIP模型: {model_name}")
            
            # 加载CLIP处理器
            self.log_info("正在加载CLIP处理器...")
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.log_info("CLIP处理器加载完成")
            
            # 加载CLIP模型
            self.log_info("正在加载CLIP模型...")
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.log_info("CLIP模型加载完成")
            
            # 设置为评估模式
            self.clip_model.eval()
            
            self.log_info(f"CLIP模型成功加载到设备: {self.device}")
            return self.clip_model, self.clip_processor
            
        except Exception as e:
            self.log_error(f"CLIP模型加载失败: {str(e)}")
            raise
    
    def load_stable_diffusion(self, model_name: str = "stabilityai/stable-diffusion-2-1-base") -> StableDiffusionPipeline:
        """
        加载Stable Diffusion 2.1模型
        
        Args:
            model_name: Stable Diffusion模型名称
            
        Returns:
            Stable Diffusion管道
        """
        try:
            self.log_info(f"开始加载Stable Diffusion模型: {model_name}")
            
            # 根据设备选择数据类型
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.log_info(f"使用数据类型: {torch_dtype}")
            
            # 加载Stable Diffusion管道
            self.log_info("正在加载Stable Diffusion管道...")
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,  # 禁用安全检查器以节省内存
                requires_safety_checker=False
            ).to(self.device)
            
            self.log_info("Stable Diffusion管道加载完成")
            
            # 启用内存优化
            if self.device == "cuda":
                self.sd_pipeline.enable_attention_slicing()
                self.sd_pipeline.enable_model_cpu_offload()
                self.log_info("已启用CUDA内存优化")
            
            self.log_info(f"Stable Diffusion模型成功加载到设备: {self.device}")
            return self.sd_pipeline
            
        except Exception as e:
            self.log_error(f"Stable Diffusion模型加载失败: {str(e)}")
            raise
    
    def load_all_models(self) -> Tuple[CLIPModel, CLIPProcessor, StableDiffusionPipeline]:
        """
        加载所有必需的模型
        
        Returns:
            (CLIP模型, CLIP处理器, Stable Diffusion管道)
        """
        self.log_info("开始加载所有模型...")
        
        # 加载CLIP模型
        clip_model, clip_processor = self.load_clip_model()
        
        # 加载Stable Diffusion模型
        sd_pipeline = self.load_stable_diffusion()
        
        self.log_info("所有模型加载完成")
        return clip_model, clip_processor, sd_pipeline
    
    def get_models(self) -> Tuple[Optional[CLIPModel], Optional[CLIPProcessor], Optional[StableDiffusionPipeline]]:
        """
        获取已加载的模型
        
        Returns:
            (CLIP模型, CLIP处理器, Stable Diffusion管道)
        """
        return self.clip_model, self.clip_processor, self.sd_pipeline
    
    def clear_models(self):
        """
        清理模型，释放内存
        """
        self.log_info("开始清理模型...")
        
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
            
        if self.clip_processor is not None:
            del self.clip_processor
            self.clip_processor = None
            
        if self.sd_pipeline is not None:
            del self.sd_pipeline
            self.sd_pipeline = None
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.log_info("模型清理完成")
    
    def get_device_info(self) -> dict:
        """
        获取设备信息
        
        Returns:
            设备信息字典
        """
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved()
            })
        
        return info
