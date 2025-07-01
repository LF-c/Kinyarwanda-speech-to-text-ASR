#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KinyaWhisper 语音转录脚本

这个脚本提供了一个简单的接口来使用KinyaWhisper模型进行卢旺达语语音转录。

使用方法:
    python kinyawhisper_transcribe.py --audio_path your_audio.wav
    python kinyawhisper_transcribe.py --audio_path your_audio.wav --output_file transcription.txt
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torchaudio
    import torch
except ImportError as e:
    print(f"错误: 缺少必要的依赖包 - {e}")
    print("请运行: pip install transformers torchaudio torch")
    sys.exit(1)


class KinyaWhisperTranscriber:
    """KinyaWhisper 转录器类"""
    
    def __init__(self, model_name="benax-rw/KinyaWhisper"):
        """
        初始化转录器
        
        Args:
            model_name (str): 模型名称，默认为 "benax-rw/KinyaWhisper"
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        try:
            print(f"正在加载模型: {self.model_name}...")
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    
    def transcribe_audio(self, audio_path):
        """
        转录音频文件
        
        Args:
            audio_path (str): 音频文件路径
            
        Returns:
            str: 转录结果
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            print(f"正在处理音频文件: {audio_path}")
            
            # 加载音频文件
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 预处理音频
            inputs = self.processor(
                waveform.squeeze(), 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            )
            
            # 生成转录
            print("正在生成转录...")
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs["input_features"])
            
            # 解码结果
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            raise RuntimeError(f"转录过程中出现错误: {e}")
    
    def transcribe_and_save(self, audio_path, output_file=None):
        """
        转录音频并保存结果
        
        Args:
            audio_path (str): 音频文件路径
            output_file (str, optional): 输出文件路径
            
        Returns:
            str: 转录结果
        """
        transcription = self.transcribe_audio(audio_path)
        
        # 如果指定了输出文件，保存结果
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"转录结果已保存到: {output_file}")
            except Exception as e:
                print(f"保存文件时出错: {e}")
        
        return transcription


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用KinyaWhisper模型进行卢旺达语语音转录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python kinyawhisper_transcribe.py --audio_path audio.wav
  python kinyawhisper_transcribe.py --audio_path audio.wav --output_file result.txt
  python kinyawhisper_transcribe.py --audio_path audio.wav --model_name your_model_name
        """
    )
    
    parser.add_argument(
        "--audio_path", 
        required=True,
        help="输入音频文件路径 (支持 .wav, .mp3, .flac 等格式)"
    )
    
    parser.add_argument(
        "--output_file", 
        help="输出文件路径 (可选，如果不指定则只在控制台显示结果)"
    )
    
    parser.add_argument(
        "--model_name", 
        default="benax-rw/KinyaWhisper",
        help="模型名称 (默认: benax-rw/KinyaWhisper)"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建转录器
        transcriber = KinyaWhisperTranscriber(model_name=args.model_name)
        
        # 执行转录
        transcription = transcriber.transcribe_and_save(
            args.audio_path, 
            args.output_file
        )
        
        # 显示结果
        print("\n" + "="*50)
        print("转录结果:")
        print("="*50)
        print(transcription)
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()