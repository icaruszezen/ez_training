# EZ Training

目标检测标注 & 训练一条龙工具，基于 PyQt5 + Fluent Design。

把 labelImg 标注、视觉 API / YOLO 预标注、模板匹配、数据准备、训练、验证这些事串到一个界面里，省得来回切工具。

## 功能一览

| 页面 | 干什么用 |
|------|----------|
| 数据集 | 建项目、扫目录、看缩略图和标注统计 |
| 预标注 | 接视觉大模型 API 或本地 YOLO 权重，批量跑 VOC 标注 |
| 标注 | 内置 labelImg，框选标注 |
| 批量标注 | 多图同时改，适合同类目标快速校正 |
| 模板匹配 | OpenCV matchTemplate，多模板、多尺度、NMS 去重 |
| 数据准备 | VOC → YOLO 格式转换，train/val 划分，可选数据增强 |
| 脚本标注 | 写 Python 脚本批量处理标注，界面上直接编辑执行 |
| 训练 | 配置 YOLO 训练参数，看实时日志 |
| 验证 | 跑 mAP / Precision / Recall / F1，导出报告 |
| 设置 | 依赖安装、镜像源等环境配置 |

## 项目结构

```
src/ez_training/
├── main.py                  # 入口
├── dep_installer.py         # 运行时依赖安装 (torch/ultralytics)
├── updater.py               # GitHub Releases 自动更新
├── ui/                      # 主窗口、绘制、后台线程
├── pages/                   # 上面表格里的各页面
├── prelabeling/             # 预标注引擎 (视觉 API + YOLO)
├── labeling/                # labelImg 集成
├── template_matching/       # 模板匹配
├── data_prep/               # 数据准备流水线
├── evaluation/              # 验证引擎 & 报告生成
├── annotation_scripts/      # 内置脚本标注示例
└── common/constants.py      # 常量、配置目录、设备检测
```

## 快速开始

### 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/ez_training/main.py
```

### 打包版（exe）

直接运行 `ez_training.exe`。首次使用需要安装 PyTorch 和 Ultralytics——运行 exe 同目录下的 `install_deps.bat`，按提示选 CUDA 版本就行。

程序会自动检查 GitHub Releases 上的新版本。

## 基本流程

1. **数据集** 建项目 → 选图片目录
2. **预标注** 跑一遍自动标注（API 或 YOLO）
3. **标注 / 批量标注** 人工过一遍，改错的补漏的
4. **数据准备** 导出 YOLO 训练集
5. **训练** 开跑
6. **验证** 看指标、导报告

## 配置文件

配置存在 `~/.ez_training/` 下：

- `datasets.json` — 数据集项目列表
- `vision_api_config.json` — 视觉 API 密钥和端点
- `settings.json` — 全局设置

## 注意

- 训练和验证需要 `ultralytics`，GPU 加速需要正确安装对应版本的 PyTorch + CUDA。
- 用视觉 API 预标注的话，先去设置里填好 endpoint 和 api_key。
- 大数据集跑模板匹配或批量任务时建议分批来，别一次塞太多。

## 技术栈

Python 3 / PyQt5 / PyQt-Fluent-Widgets / OpenCV / Ultralytics YOLO / Albumentations / lxml / Matplotlib
