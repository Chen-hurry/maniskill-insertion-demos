cat > data/README.md <<'EOF'
# Data Directory / 数据目录说明

This directory stores trajectories, RGB frames, videos, and labels used by the robust recovery experiments.

本目录用于保存鲁棒恢复实验中的轨迹数据、RGB 图像帧、视频以及标签信息。

## Directory Layout / 目录结构

```text
data/
├── nominal/      # Normal executions without injected anomalies / 正常执行数据：未注入异常的任务轨迹
├── anomalies/    # Executions with injected anomalies / 异常执行数据：包含遮挡、物体偏移、轨迹偏离等异常
├── recovery/     # Recovery rollouts after anomaly or failure states / 恢复数据：从异常或失败状态开始的恢复轨迹
└── demos/        # Expert, scripted, or human-corrected demonstrations / 示范数据：专家示范、脚本策略或人工纠正轨迹