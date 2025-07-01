#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KinyaWhisper 使用示例

这个文件展示了如何使用 kinyawhisper_transcribe.py 脚本进行语音转录。
"""

import os
from kinyawhisper_transcribe import KinyaWhisperTranscriber

def example_usage():
    """示例用法"""
    
    # 检查是否有音频文件用于测试
    # 您可以将这里的路径替换为您的实际音频文件路径
    audio_files = [
        "test_audio.wav",
        "sample.wav", 
        "audio.wav"
    ]
    
    # 寻找可用的音频文件
    test_audio = None
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            test_audio = audio_file
            break
    
    if not test_audio:
        print("未找到测试音频文件。")
        print("请将您的音频文件命名为以下之一并放在当前目录：")
        for audio_file in audio_files:
            print(f"  - {audio_file}")
        print("\n或者修改此脚本中的 audio_files 列表以包含您的音频文件路径。")
        return
    
    try:
        print("=" * 60)
        print("KinyaWhisper 转录示例")
        print("=" * 60)
        
        # 创建转录器实例
        print("正在初始化转录器...")
        transcriber = KinyaWhisperTranscriber()
        
        # 方法1: 仅转录
        print(f"\n方法1: 转录音频文件 '{test_audio}'")
        transcription = transcriber.transcribe_audio(test_audio)
        print(f"转录结果: {transcription}")
        
        # 方法2: 转录并保存到文件
        output_file = "transcription_result.txt"
        print(f"\n方法2: 转录并保存到 '{output_file}'")
        transcription = transcriber.transcribe_and_save(test_audio, output_file)
        print(f"转录结果: {transcription}")
        
        print("\n" + "=" * 60)
        print("转录完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保：")
        print("1. 已安装所需依赖: pip install transformers torchaudio torch")
        print("2. 网络连接正常（首次使用需要下载模型）")
        print("3. 音频文件格式正确且可读")

if __name__ == "__main__":
    example_usage()