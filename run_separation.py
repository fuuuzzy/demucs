#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demucs 音频分离脚本
支持单个文件或批量处理多个音频文件
"""

import argparse
import sys
from pathlib import Path
import torch

# 检查是否安装了 demucs
try:
    from demucs.api import Separator, save_audio
    from demucs.pretrained import get_model
except ImportError:
    print("错误: 未找到 demucs 模块")
    print("请先安装: uv pip install -e .")
    sys.exit(1)


def check_cuda_available():
    """检查 CUDA 是否可用"""
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用 - 使用 GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠ CUDA 不可用 - 使用 CPU 模式")
        return False


def separate_audio(
    input_path,
    output_dir="separated",
    model_name="htdemucs",
    device=None,
    shifts=1,
    overlap=0.25,
    split=True,
    segment=None,
    jobs=0,
):
    """
    分离音频文件
    
    参数:
        input_path: 输入音频文件路径
        output_dir: 输出目录
        model_name: 模型名称 (htdemucs, htdemucs_ft, mdx_extra, etc.)
        device: 设备 ('cuda' 或 'cpu')
        shifts: 随机偏移次数，提高质量但增加时间
        overlap: 分段重叠比例
        split: 是否分段处理（节省内存）
        segment: 分段大小（秒）
        jobs: 并行任务数
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    if not input_path.exists():
        print(f"错误: 文件不存在 - {input_path}")
        return False
    
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"偏移次数: {shifts}")
    print(f"{'='*60}\n")
    
    try:
        # 创建分离器
        separator = Separator(
            model=model_name,
            device=device,
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment,
            jobs=jobs,
        )
        
        print(f"正在处理: {input_path.name}...")
        
        # 执行分离
        origin, separated = separator.separate_audio_file(input_path)
        
        # 保存分离后的音轨
        output_path = output_dir / input_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取音轨名称
        stems = list(separated.keys())
        print(f"\n分离出的音轨: {', '.join(stems)}")
        
        for stem_name, stem_audio in separated.items():
            output_file = output_path / f"{stem_name}.wav"
            save_audio(stem_audio, output_file, samplerate=separator.samplerate)
            print(f"✓ 已保存: {output_file}")
        
        print(f"\n✓ 完成! 输出目录: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_separate(
    input_dir,
    output_dir="separated",
    model_name="htdemucs",
    device=None,
    extensions=None,
    **kwargs
):
    """
    批量处理目录中的音频文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        model_name: 模型名称
        device: 设备
        extensions: 支持的文件扩展名列表
        **kwargs: 其他参数传递给 separate_audio
    """
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus']
    
    input_dir = Path(input_dir)
    
    if not input_dir.is_dir():
        print(f"错误: 不是有效的目录 - {input_dir}")
        return
    
    # 查找所有音频文件
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"错误: 在 {input_dir} 中未找到音频文件")
        print(f"支持的格式: {', '.join(extensions)}")
        return
    
    print(f"\n找到 {len(audio_files)} 个音频文件")
    
    success_count = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] 处理: {audio_file.name}")
        if separate_audio(audio_file, output_dir, model_name, device, **kwargs):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"批量处理完成: {success_count}/{len(audio_files)} 成功")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Demucs 音频分离工具 - 支持单文件或批量处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分离单个文件
  python run_separation.py song.mp3
  
  # 分离单个文件到指定目录
  python run_separation.py song.mp3 -o output
  
  # 使用不同的模型
  python run_separation.py song.mp3 -m htdemucs_ft
  
  # 批量处理目录中的所有音频文件
  python run_separation.py -b music_folder/
  
  # 强制使用 CPU
  python run_separation.py song.mp3 --cpu
  
  # 提高质量（增加偏移次数）
  python run_separation.py song.mp3 --shifts 10

可用模型:
  - htdemucs (默认): 4 stems (drums, bass, other, vocals)
  - htdemucs_ft: Fine-tuned version
  - htdemucs_6s: 6 stems (drums, bass, other, vocals, guitar, piano)
  - mdx_extra: MDX-Net model
  - mdx_extra_q: Quantized MDX-Net
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="输入音频文件或目录（批量模式）"
    )
    
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="批量处理模式（处理目录中的所有音频文件）"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="separated",
        help="输出目录 (默认: separated)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="htdemucs",
        help="模型名称 (默认: htdemucs)"
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="设备选择 (默认: 自动检测)"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU"
    )
    
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="随机偏移次数，提高质量但增加处理时间 (默认: 1, 推荐: 5-10)"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="分段重叠比例 (默认: 0.25)"
    )
    
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="不分段处理（需要更多内存）"
    )
    
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="分段大小（秒），None 为自动"
    )
    
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="并行任务数 (默认: 0 = 自动)"
    )
    
    parser.add_argument(
        "--check-cuda",
        action="store_true",
        help="检查 CUDA 是否可用并退出"
    )
    
    args = parser.parse_args()
    
    # 检查 CUDA
    if args.check_cuda:
        check_cuda_available()
        return
    
    if not args.input:
        parser.print_help()
        return
    
    # 设备选择
    device = "cpu" if args.cpu else args.device
    
    # 显示 CUDA 状态
    check_cuda_available()
    
    # 批量处理或单文件处理
    if args.batch:
        batch_separate(
            input_dir=args.input,
            output_dir=args.output,
            model_name=args.model,
            device=device,
            shifts=args.shifts,
            overlap=args.overlap,
            split=not args.no_split,
            segment=args.segment,
            jobs=args.jobs,
        )
    else:
        separate_audio(
            input_path=args.input,
            output_dir=args.output,
            model_name=args.model,
            device=device,
            shifts=args.shifts,
            overlap=args.overlap,
            split=not args.no_split,
            segment=args.segment,
            jobs=args.jobs,
        )


if __name__ == "__main__":
    main()

