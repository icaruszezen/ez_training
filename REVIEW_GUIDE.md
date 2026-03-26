# EZ Traing 功能 Review 指南

> 本文档为项目各功能模块提供 AI Code Review 提示词，可直接用于 AI 辅助审查。
> 项目版本：0.2.6 | 技术栈：Python 3 / PyQt5 / PyQt-Fluent-Widgets / Ultralytics YOLO / OpenCV

---

## 目录

1. [数据集管理](#1-数据集管理)
2. [预标注](#2-预标注)
3. [图像标注（labelImg 集成）](#3-图像标注labelimg-集成)
4. [批量标注](#4-批量标注)
5. [模板匹配](#5-模板匹配)
6. [数据准备](#6-数据准备)
7. [脚本标注](#7-脚本标注)
8. [模型训练](#8-模型训练)
9. [模型验证/评估](#9-模型验证评估)
10. [小工具（YOLO→VOC 转换）](#10-小工具yolovoc-转换)
11. [标注指导文档导出](#11-标注指导文档导出)
12. [设置页面](#12-设置页面)
13. [自动更新](#13-自动更新)
14. [依赖安装](#14-依赖安装)
15. [UI 主窗口与页面框架](#15-ui-主窗口与页面框架)
16. [通用模块](#16-通用模块)

---

## 1. 数据集管理

**涉及文件：**
- `src/ez_training/pages/dataset_page.py`（~2268 行）
- `src/ez_training/ui/workers.py`（ThumbnailLoader, ImageScanWorker）

**核心功能：** 项目 CRUD、归档管理、子文件夹模式、图片列表（分页/筛选/缩略图）、标注统计、标注预览、跳转到标注/批量标注页面。

### Review 提示词

```
请对以下数据集管理模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/dataset_page.py
- src/ez_training/ui/workers.py

请重点审查以下方面：

1. **ProjectManager 持久化**
   - datasets.json 的读写是否有竞态条件风险（多线程环境下）？
   - 项目增删改操作是否每次都完整序列化？是否存在部分写入导致 JSON 损坏的风险？
   - 是否需要备份机制或原子写入？

2. **ImageScanner 线程安全**
   - QThread 扫描图片和标注时，是否正确使用信号槽传递数据？
   - 大量图片（>10000 张）时的内存占用和扫描耗时是否合理？
   - 取消扫描时资源是否正确释放？

3. **分页与缩略图加载**
   - 200 张/页的分页策略是否合理？切换页面时旧缩略图是否及时释放？
   - ThumbnailLoader 是否会在快速翻页时产生过多排队任务？是否有取消旧请求的机制？
   - 缩略图缓存策略是否得当（内存上限、LRU 淘汰等）？

4. **标注统计准确性**
   - VOC XML 和 YOLO TXT 标注的统计逻辑是否一致？
   - 是否正确处理了无标注文件、空标注文件、格式错误文件？
   - AnnotationStats 的标签分布计算对大数据集是否有性能问题？

5. **归档与子文件夹模式**
   - 创建/解散归档时数据一致性如何保证？
   - 子文件夹模式下，嵌套目录结构是否被正确遍历？
   - 边界情况：空目录、权限不足、符号链接、中文路径。

6. **UI 响应性**
   - 长时间扫描操作是否会阻塞 UI？
   - 标注预览（ImagePreviewWidget）绘制标注框的性能如何？
   - 列表筛选（已标注/未标注/类型）的实时性是否满足要求？

7. **信号机制**
   - request_annotation 和 request_batch_annotation 信号传递的参数是否完整且类型安全？
   - 页面间数据传递是否存在过时数据的问题？
```

---

## 2. 预标注

**涉及文件：**
- `src/ez_training/pages/prelabeling_page.py`
- `src/ez_training/prelabeling/engine.py`
- `src/ez_training/prelabeling/vision_service.py`
- `src/ez_training/prelabeling/yolo_service.py`
- `src/ez_training/prelabeling/voc_writer.py`
- `src/ez_training/prelabeling/config.py`
- `src/ez_training/prelabeling/models.py`

**核心功能：** 支持视觉大模型 API 和本地 YOLO 两种后端进行自动预标注，生成 VOC 格式标注文件。

### Review 提示词

```
请对以下预标注模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/prelabeling_page.py
- src/ez_training/prelabeling/ 目录下所有文件

请重点审查以下方面：

1. **VisionModelService API 调用**
   - API Key 的存储和传输是否安全？是否有明文泄露风险？
   - 请求超时和重试机制是否完善？网络中断时的错误处理是否友好？
   - Base64 编码大图时的内存消耗是否可控？
   - _extract_json() 从 Markdown 代码块提取 JSON 的鲁棒性如何？能否处理各种 LLM 输出格式？
   - parse_response() 对异常格式（缺少字段、坐标越界、负值等）的容错能力。

2. **YoloModelService 本地推理**
   - 模型加载是否有缓存机制，避免重复加载？
   - GPU/CPU 设备选择逻辑是否正确？OOM 时是否有降级方案？
   - 推理结果到 BoundingBox 的坐标转换是否准确？

3. **PrelabelingWorker 并发处理**
   - ThreadPoolExecutor 的并发数配置是否合理？
   - 并发调用 API 时是否有速率限制保护？
   - 线程异常是否被正确捕获并上报？
   - 取消操作（用户中断）时资源清理是否完整？

4. **VOCAnnotationWriter**
   - 合并已有标注的逻辑是否正确（不丢失手动标注）？
   - 图片尺寸缓存（LRU）策略是否合理？
   - 坐标边界裁剪（clamp to image bounds）是否正确实现？
   - 空检测结果是否会覆盖已有标注？

5. **APIConfigManager**
   - 配置文件路径的跨平台兼容性。
   - 敏感信息（API Key）在磁盘上的存储安全性。
   - 配置文件损坏时的恢复能力。

6. **检测模式切换**
   - "仅文本提示"和"文本+参考图"两种模式的 UI 状态切换是否完整？
   - 参考图数量限制（最多 10 张）的校验是否到位？
   - 模式切换后之前的配置是否正确保留？

7. **整体流程**
   - 跳过已标注图片的逻辑是否可配置？
   - 进度报告是否准确（已处理/总数/跳过数）？
   - 大批量图片（>1000 张）时的稳定性和内存占用。
```

---

## 3. 图像标注（labelImg 集成）

**涉及文件：**
- `src/ez_training/pages/annotation_page.py`
- `src/ez_training/labeling/annotation_window.py`
- `src/ez_training/labeling/label_app.py`
- `src/ez_training/labeling/canvas.py`
- `src/ez_training/labeling/shape.py`
- `src/ez_training/labeling/label_file.py`
- `src/ez_training/labeling/pascal_voc_io.py`
- `src/ez_training/labeling/yolo_io.py`
- `src/ez_training/labeling/create_ml_io.py`

**核心功能：** 集成 labelImg 标注工具，支持 PascalVOC/YOLO/CreateML 格式，提供画框、编辑、缩放、亮度调节等功能。

### Review 提示词

```
请对以下标注模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/annotation_page.py
- src/ez_training/labeling/ 目录下所有文件

请重点审查以下方面：

1. **Canvas 交互逻辑**
   - 画框、移动、调整大小的鼠标事件处理是否流畅？
   - 坐标从屏幕坐标到图片坐标的转换是否精确？缩放后是否有精度损失？
   - snap_point_to_canvas 边界裁剪在极端缩放比例下是否正确？
   - 多个重叠标注框的选择逻辑（nearest_vertex、contains_point）是否合理？

2. **Shape 数据模型**
   - 标注框的数据结构是否适合后续扩展（如多边形、旋转框）？
   - bounding_rect() 和 make_path() 在极小框（宽或高 < 1px）时的表现？

3. **多格式 IO 一致性**
   - PascalVOC、YOLO、CreateML 三种格式之间的读写转换是否可逆且无损？
   - YOLO 格式的归一化坐标 ↔ 像素坐标转换精度如何？
   - 标签中包含特殊字符（空格、中文、引号）时各格式是否正确处理？
   - classes.txt 的读写在多次保存后是否保持类别顺序一致？

4. **AnnotationWindow Fluent 封装**
   - 继承 MainWindow 时是否有方法覆盖遗漏或行为冲突？
   - 预设标签编辑（PredefinedLabelsDialog）的保存和加载是否可靠？
   - 快捷键 T（改标签）在各种状态下是否稳定？
   - 复制/粘贴标注功能的坐标处理是否正确？

5. **亮度和缩放**
   - 亮度覆盖层的实现对大图的渲染性能影响。
   - fit_window / fit_width / manual 三种缩放模式切换是否平滑？
   - 缩放极值（极大/极小）时 UI 是否仍然可用？

6. **文件管理**
   - 自动保存和手动保存的冲突处理。
   - 切换图片时未保存修改的提醒机制是否完善？
   - 文件列表的自然排序在各种文件名格式下是否正确？

7. **多语言支持**
   - StringBundle 加载 zh-CN / ja-JP / zh-TW 的回退机制。
   - 多语言字符串中缺失的 key 的处理。
```

---

## 4. 批量标注

**涉及文件：**
- `src/ez_training/pages/batch_annotation_page.py`

**核心功能：** 在第一张图上标注后，将标注批量应用到同分辨率的其他图片；支持加样（将图片复制到指定数据集目录）。

### Review 提示词

```
请对以下批量标注模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/batch_annotation_page.py

请重点审查以下方面：

1. **批量应用逻辑（BatchApplyWorker）**
   - 仅对同分辨率图片应用标注的策略是否合理？是否需要按比例缩放标注？
   - 合并已有标注时的去重逻辑（_shape_bbox_key）是否充分？相近但不完全相同的框如何处理？
   - _read_existing_voc_shapes 和 _read_existing_yolo_shapes 读取已有标注的兼容性。
   - 批量写入过程中断（异常/用户取消）时已写入的文件是否一致？

2. **分辨率匹配**
   - 图片分辨率获取是否高效（是否需要完整加载图片）？
   - 同一批次中分辨率不一致的提示是否清晰（红色标记）？
   - 极端情况：EXIF 旋转标记导致的宽高互换是否被考虑？

3. **加样功能**
   - 加样图片复制到 sample_dataset_dir 时是否检查目标路径冲突？
   - 同名文件的处理策略（覆盖/跳过/重命名）？
   - 标注文件是否随图片一起复制？

4. **UI 交互**
   - 左侧标注编辑与右侧列表的联动是否流畅？
   - 多选操作（全选/反选）对大量图片的性能。
   - 成功/失败/不匹配三种状态的颜色标识是否清晰直观？

5. **线程安全**
   - BatchApplyWorker 运行期间 UI 交互的安全性。
   - 进度信号和完成信号的发射时序是否正确？
```

---

## 5. 模板匹配

**涉及文件：**
- `src/ez_training/pages/template_matching_page.py`
- `src/ez_training/pages/template_editor_dialog.py`
- `src/ez_training/template_matching/matcher.py`
- `src/ez_training/template_matching/worker.py`

**核心功能：** 使用 OpenCV 模板匹配在图片中查找目标，支持多模板、多尺度、预处理、NMS 去重，并可保存为 VOC 标注。

### Review 提示词

```
请对以下模板匹配模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/template_matching_page.py
- src/ez_training/pages/template_editor_dialog.py
- src/ez_training/template_matching/matcher.py
- src/ez_training/template_matching/worker.py

请重点审查以下方面：

1. **TemplateMatcher 核心算法**
   - matchTemplate 方法（TM_CCOEFF_NORMED 等）的选择是否合适？SQDIFF 系列的阈值方向是否正确处理？
   - 多尺度搜索的缩放范围和步长配置是否合理？性能如何？
   - NMS 去重（cv2.dnn.NMSBoxes）的 IoU 阈值是否可配置？
   - 模板大于目标图时的防御处理。

2. **预处理管道（PreprocessConfig）**
   - 灰度、模糊、二值化、Canny 各步骤的参数选择依据。
   - 预处理链的顺序是否合理？是否存在无效组合？
   - 预处理后 matchTemplate 的精度变化是否在预期范围内？

3. **TemplateMatchingWorker 批量处理**
   - 大量图片（>1000 张）的内存管理：每张图是否及时释放？
   - 跳过已标注的逻辑是否正确？
   - 进度报告的粒度是否足够？

4. **TemplateEditorDialog 模板编辑**
   - CropImageWidget 的裁剪区域选取是否精确？
   - 缩放、平移操作的流畅度。
   - 匹配测试结果的可视化是否清晰？
   - ROI 限定区域的交互逻辑是否直观？

5. **结果审查与保存**
   - 匹配结果表格中的勾选/排除操作是否便捷？
   - 批量保存 VOC 时是否正确合并已有标注？
   - 快速标定（QuickAnnotateDialog）手动绘框的交互体验。

6. **中文路径兼容性**
   - imread_unicode 是否在所有场景下被正确使用？
   - 临时文件的中文路径处理。

7. **加样与复制**
   - 加样逻辑与批量标注的加样是否复用？是否一致？
```

---

## 6. 数据准备

**涉及文件：**
- `src/ez_training/pages/data_prep_page.py`
- `src/ez_training/data_prep/pipeline.py`
- `src/ez_training/data_prep/converter.py`
- `src/ez_training/data_prep/splitter.py`
- `src/ez_training/data_prep/augmentation.py`
- `src/ez_training/data_prep/models.py`

**核心功能：** VOC→YOLO 转换、训练/验证集划分（防泄露）、数据增强（17 种方法）、生成 data.yaml。

### Review 提示词

```
请对以下数据准备模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/data_prep_page.py
- src/ez_training/data_prep/ 目录下所有文件

请重点审查以下方面：

1. **数据泄露控制（splitter.py）**
   - _normalize_base_stem() 的正则是否覆盖所有常见增强后缀（aug、flip、rot 等）？
   - _leakage_group_key() 的分组策略是否充分防止泄露？
   - 同源样本识别是否过于激进（误将不同样本归为同组）或过于保守？
   - 训练/验证比例在极端情况下的表现（如只有 1-2 张图片）。

2. **VOC→YOLO 转换（converter.py）**
   - VOC XML 解析是否处理了所有标准字段和非标准扩展？
   - 坐标归一化精度（像素→归一化→像素）的误差累积。
   - 缓存机制（clear_voc_cache）的生命周期管理。
   - classes.txt 的类别顺序在多次转换中是否保持一致？
   - 同名图片不同格式（如 img.jpg 和 img.png）的歧义处理。

3. **数据增强（augmentation.py）**
   - 17 种增强方法的参数范围是否适用于目标检测任务？
   - 增强后标注框坐标是否正确变换（翻转、旋转等几何变换后的框变换）？
   - build_augmenter() 组合多种增强时的概率计算是否正确？
   - Albumentations 不可用时的降级处理。

4. **DataPrepPipeline 整体流程**
   - 扫描→划分→导出→增强→写 data.yaml 的事务性如何保证？中途失败是否清理？
   - 线程池并行增强时的线程数选择是否合理？
   - 大数据集（>10000 图）时的内存峰值估算。
   - data.yaml 中的路径格式（绝对/相对）是否跨平台兼容？

5. **UI 状态持久化**
   - data_prep_ui_state.json 的读写是否与主配置文件冲突？
   - 增强方法的选择状态在重启后是否正确恢复？

6. **边界情况**
   - 空数据集（0 张图片）的处理。
   - 只有标注没有图片（或只有图片没有标注）的情况。
   - 输出目录已存在且非空时的覆盖策略。
   - 自定义 classes.txt 与实际标注不匹配时的处理。
```

---

## 7. 脚本标注

**涉及文件：**
- `src/ez_training/pages/script_annotation_page.py`
- `src/ez_training/annotation_scripts/voc_utils.py`

**核心功能：** 用户编写 Python 脚本进行自动标注，通过 QProcess 子进程运行，提供脚本管理和日志查看。

### Review 提示词

```
请对以下脚本标注模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/script_annotation_page.py
- src/ez_training/annotation_scripts/voc_utils.py

请重点审查以下方面：

1. **安全性（最高优先级）**
   - 用户脚本通过 QProcess 执行，是否存在代码注入风险？
   - dataset_dir 参数传递是否有路径遍历风险？
   - 脚本运行权限是否受限？是否能访问敏感文件？
   - 是否需要沙箱隔离机制？

2. **脚本生命周期管理**
   - 新建、刷新、删除脚本的文件操作是否有竞态条件？
   - 脚本编辑后的保存和重新加载是否可靠？
   - 脚本文件编码（UTF-8）处理是否正确？

3. **QProcess 子进程管理**
   - 进程启动失败的错误处理。
   - 停止按钮是否能可靠终止子进程（包括其子进程）？
   - 长时间运行脚本的超时机制。
   - 进程输出（stdout/stderr）的实时显示是否完整？
   - 进程退出码的解读和展示。

4. **voc_utils 工具函数**
   - create_voc_root 生成的 XML 结构是否符合 PascalVOC 标准？
   - read_existing_objects 对损坏 XML 的容错能力。
   - save_voc 的原子写入问题。
   - run_annotation 的"扫描→检测→保存"骨架是否足够灵活？

5. **脚本约定**
   - run(dataset_dir) 的约定是否在 UI 中清晰说明？
   - 脚本模板或示例是否足够帮助用户上手？
   - 脚本依赖管理（如需要额外 pip 包）是否被考虑？
```

---

## 8. 模型训练

**涉及文件：**
- `src/ez_training/pages/train_page.py`

**核心功能：** 配置和启动 YOLOv8 训练，支持 n/s/m/l/x 模型选择、超参数配置、日志流式显示、训练记录和权重管理。

### Review 提示词

```
请对以下训练模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/train_page.py

请重点审查以下方面：

1. **YoloTrainThread 训练线程**
   - 训练在 QThread 中调用 ultralytics，是否存在 GIL 或信号槽的线程安全问题？
   - 训练回调获取 epoch 进度的实现是否正确且实时？
   - 训练异常（OOM、数据错误、模型加载失败）的捕获和上报。
   - 停止训练时 YOLO 的清理是否完整（GPU 显存释放）？

2. **配置验证**
   - data.yaml 路径是否被验证存在且格式正确？
   - Epochs、Batch Size、Image Size 的取值范围校验。
   - Device 选择（CPU/GPU）与实际可用设备的一致性检查。
   - 输出目录的权限和空间检查。

3. **日志管理**
   - LogPanel 的日志显示在长时间训练时的性能（日志行数增长）。
   - 是否有日志截断或滚动机制？
   - ANSI 转义序列的清洗（strip_ansi）是否完整？

4. **训练记录（WeightPanel）**
   - 训练记录的发现和展示逻辑是否正确？
   - 权重文件列表是否包含了 best.pt 和 last.pt？
   - 打开目录功能的跨平台兼容性。

5. **资源管理**
   - 训练期间 UI 的响应性。
   - 多次启动/停止训练是否会导致资源泄漏？
   - GPU 显存在训练结束后是否完全释放？

6. **用户体验**
   - 训练前的预检查（数据集是否就绪、模型是否可下载）。
   - 训练进度百分比的准确性。
   - 训练结束后的结果总结是否清晰？
```

---

## 9. 模型验证/评估

**涉及文件：**
- `src/ez_training/pages/eval_page.py`
- `src/ez_training/evaluation/engine.py`
- `src/ez_training/evaluation/models.py`
- `src/ez_training/evaluation/visualization.py`
- `src/ez_training/evaluation/report_generator.py`

**核心功能：** 使用 YOLO val 进行验证，展示 mAP/Precision/Recall/F1 指标，生成图表，导出 JSON/CSV 报告。

### Review 提示词

```
请对以下验证/评估模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/eval_page.py
- src/ez_training/evaluation/ 目录下所有文件

请重点审查以下方面：

1. **EvaluationEngine 核心逻辑**
   - build_data_yaml() 生成的 data.yaml 路径和类别是否总是正确？
   - YOLO val 调用的参数传递是否完整？
   - mAP、Precision、Recall、F1 的提取逻辑是否准确？
   - _safe_float() 对 NaN/Inf 的处理是否充分？
   - _sanitize_name() 的清洗规则是否过于激进？

2. **数据集适配**
   - 从 ProjectManager 获取数据集目录后，classes.txt 的查找逻辑。
   - 数据集格式（需要 YOLO 格式）的前提条件校验。
   - 模型的类别数与数据集类别数不匹配时的处理。

3. **可视化（visualization.py）**
   - discover_yolo_plots() 查找 YOLO 生成图表的路径是否覆盖所有 YOLO 版本？
   - generate_fallback_charts() 的 matplotlib 图表质量和可读性。
   - 图表的中文标签显示是否正确（matplotlib 字体配置）？

4. **报告导出（report_generator.py）**
   - export_reports() 导出 JSON 和 CSV 的数据完整性。
   - 浮点数精度格式化是否统一？
   - 导出路径冲突（文件已存在）的处理。

5. **UI 交互**
   - EvalResultPanel 的指标展示是否清晰直观？
   - ClickableImageLabel 点击打开原图的可靠性。
   - 复制指标到剪贴板的格式是否便于使用？
   - 另存报告的文件对话框交互。

6. **线程安全**
   - YoloEvalThread 中 YOLO val 的异常处理。
   - 验证过程中取消操作的安全性。
   - 验证结束后 GPU 资源释放。
```

---

## 10. 小工具（YOLO→VOC 转换）

**涉及文件：**
- `src/ez_training/pages/tools_page.py`

**核心功能：** 将 YOLO 格式标注转换为 VOC XML 格式。

### Review 提示词

```
请对以下工具模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/tools_page.py

请重点审查以下方面：

1. **_YoloToVocWorker 转换逻辑**
   - YOLO 归一化坐标到 VOC 像素坐标的转换精度。
   - classes.txt 中类别索引与 YOLO 标签中索引的对应关系校验。
   - 索引越界（标签索引 > classes.txt 行数）的处理。
   - 图片复制操作的效率（大文件、大批量）。

2. **输入验证**
   - 图片目录、classes.txt、标注目录、输出目录的路径校验。
   - 输入目录为空或不存在图片时的处理。
   - classes.txt 格式校验（空行、空格、编码）。

3. **错误处理**
   - 单张图片转换失败时是否影响整体进度？
   - 日志是否记录了足够的错误信息？
   - 取消操作的及时性。

4. **可扩展性**
   - ToolsPage 的 CardWidget 布局是否便于添加更多工具？
   - 工具间是否共享通用的进度/日志机制？
```

---

## 11. 标注指导文档导出

**涉及文件：**
- `src/ez_training/pages/annotation_guide_page.py`

**核心功能：** 扫描标注数据，按标签抽样裁剪图片，生成包含裁剪图和源信息的 Excel 标注指导文档。

### Review 提示词

```
请对以下标注指导模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/annotation_guide_page.py

请重点审查以下方面：

1. **LabelScanWorker 标注扫描**
   - 同时支持 VOC XML 和 YOLO TXT 的解析是否一致？
   - classes.txt 加载逻辑的优先级和回退。
   - 扫描大量文件时的内存占用（BBoxRecord 列表）。

2. **GuideExportWorker 导出逻辑**
   - 按标签抽样是否保证随机性和代表性？
   - 截取扩展比例（expand ratio）的坐标计算是否正确处理边界？
   - 画框后的裁剪图质量和可辨识度。
   - openpyxl 插入图片的内存消耗（大量高分辨率裁剪图）。

3. **Excel 文件质量**
   - 生成的 Excel 文件是否在不同版本 Excel/WPS 中可正常打开？
   - 列宽、行高是否自适应图片大小？
   - 大量标签（>50 类）时的 Excel 页面布局。

4. **UI 交互**
   - 多选数据集（含归档）的选择逻辑是否清晰？
   - 全选/全不选标签的交互是否便捷？
   - 进度反馈是否充分？

5. **性能**
   - 图片读取和裁剪在数千张图片时的耗时。
   - 是否有不必要的重复图片加载？
```

---

## 12. 设置页面

**涉及文件：**
- `src/ez_training/pages/settings_page.py`

**核心功能：** 应用版本与更新、GitHub 加速镜像、加样设置、环境信息显示、依赖安装。

### Review 提示词

```
请对以下设置模块进行全面 Code Review：

涉及文件：
- src/ez_training/pages/settings_page.py

请重点审查以下方面：

1. **更新机制**
   - 版本检查的触发频率是否合理？
   - 下载过程中断的恢复能力。
   - 自动更新（frozen 模式）的安全性：是否验证下载包的完整性（校验和）？
   - 更新失败后的回滚能力。

2. **GitHub 加速镜像**
   - 预设镜像列表的维护和有效性。
   - 自定义 URL 的格式校验。
   - 镜像对 API 和下载的不同影响是否被区分？

3. **环境信息**
   - _get_package_info() 在包未安装时的异常处理。
   - GPU 信息获取在无 CUDA 环境下的表现。
   - 信息刷新的时机（仅在页面打开时还是可手动刷新）。

4. **依赖安装（DepsInstallCard）**
   - 一键安装的 pip 命令是否正确？
   - Python 路径检测的准确性。
   - 安装过程的日志输出是否实时？
   - 安装失败的错误提示是否有帮助？

5. **设置持久化**
   - load_settings / save_settings 的调用时机是否合理？
   - 多个设置项同时修改时是否有保存冲突？
   - 设置文件损坏时的默认值回退。
```

---

## 13. 自动更新

**涉及文件：**
- `src/ez_training/updater.py`

**核心功能：** 通过 GitHub Releases API 检查新版本、下载 zip 更新包、替换并重启。

### Review 提示词

```
请对以下自动更新模块进行全面 Code Review：

涉及文件：
- src/ez_training/updater.py

请重点审查以下方面：

1. **安全性**
   - 下载的 zip 文件是否验证完整性（SHA256 校验和）？
   - Zip 路径安全检查是否能防止 Zip Slip 攻击（路径穿越）？
   - HTTPS 证书验证是否启用？
   - 更新脚本（bat）的权限问题。

2. **版本比较**
   - _compare_versions() 的版本号解析是否处理了预发布版本（alpha、beta、rc）？
   - 版本号格式不标准时的容错。

3. **下载可靠性**
   - 网络中断后的断点续传能力。
   - 下载进度报告的准确性。
   - 超时设置是否合理？
   - GitHub API 速率限制的处理。

4. **更新应用（apply_update_and_restart）**
   - 替换文件时的原子性：是否先备份旧版本？
   - 替换失败后的回滚。
   - 重启过程中用户数据是否安全？
   - 非 frozen 环境下调用更新的防御处理。

5. **镜像支持**
   - _mirror_url() 的 URL 转换逻辑是否覆盖所有 GitHub URL 模式？
   - 镜像不可用时的回退到原始 URL。
```

---

## 14. 依赖安装

**涉及文件：**
- `src/ez_training/dep_installer.py`

**核心功能：** 在 PyInstaller 打包版中安装 torch 和 ultralytics 到 deps/ 目录。

### Review 提示词

```
请对以下依赖安装模块进行全面 Code Review：

涉及文件：
- src/ez_training/dep_installer.py

请重点审查以下方面：

1. **InstallWorker 安装流程**
   - pip install --target 的兼容性（不同 pip 版本的行为差异）。
   - 安装目标目录的空间检查。
   - 安装过程中的网络错误处理。
   - 部分安装（已安装部分包后失败）时的清理。

2. **Python 环境检测**
   - find_system_python() 在不同 Windows 安装方式下是否都能找到 Python？
   - Python 版本一致性检查的重要性和实现。
   - 虚拟环境场景下的表现。

3. **_DepsFinder（main.py 中）**
   - sys.meta_path 拦截的优先级和副作用。
   - deps/ 目录中包的版本与应用其他依赖的兼容性。
   - 动态加载可能导致的导入顺序问题。

4. **用户体验**
   - 安装进度的实时反馈。
   - 安装命令是否清晰展示以便用户手动执行？
   - CUDA 版本选择（11.8 / 12.1）的建议是否准确？
```

---

## 15. UI 主窗口与页面框架

**涉及文件：**
- `src/ez_training/ui/main_window.py`
- `src/ez_training/ui/painting.py`
- `src/ez_training/ui/workers.py`

**核心功能：** FluentWindow 主窗口、延迟页面加载、ProjectManager 共享、通用绘制和后台线程。

### Review 提示词

```
请对以下 UI 框架模块进行全面 Code Review：

涉及文件：
- src/ez_training/ui/main_window.py
- src/ez_training/ui/painting.py
- src/ez_training/ui/workers.py

请重点审查以下方面：

1. **AppWindow 架构**
   - 12+ 个子页面的注册和管理是否清晰？
   - ProjectManager 的共享方式是否容易引入耦合？
   - 导航栏的图标和文案是否统一规范？

2. **LazyPageHost 延迟加载**
   - ensure_page() 的首次创建是否线程安全？
   - 延迟加载页面的信号连接时机是否正确（在页面创建后才连接）？
   - 内存释放：已创建但长时间未使用的页面是否有回收机制？

3. **页面跳转**
   - request_annotation / request_batch_annotation 信号触发页面跳转时的数据传递完整性。
   - 跳转目标页面尚未初始化时的处理。
   - 快速连续跳转是否稳定？

4. **painting.py 绘制函数**
   - draw_box_label() 的中文字体兼容性。
   - 绘制大量标注框时的性能。
   - begin_label_painter() 的 QPainter 状态管理。

5. **workers.py 后台线程**
   - ThumbnailLoader 的线程终止和重启是否安全？
   - ImageScanWorker 对符号链接、隐藏文件的处理。
   - SUPPORTED_IMAGE_FORMATS 是否覆盖常见格式？

6. **整体 UI 质量**
   - Fluent Design 风格的一致性。
   - 窗口大小、布局在不同分辨率/DPI 下的适配。
   - 异常弹窗（InfoBar、MessageBox）的风格统一性。
```

---

## 16. 通用模块

**涉及文件：**
- `src/ez_training/common/constants.py`
- `src/ez_training/__init__.py`

**核心功能：** 配置目录管理、设置读写、设备检测、ANSI 清洗、GitHub 镜像、路径打开。

### Review 提示词

```
请对以下通用模块进行全面 Code Review：

涉及文件：
- src/ez_training/common/constants.py
- src/ez_training/__init__.py

请重点审查以下方面：

1. **配置管理**
   - get_config_dir() 创建 ~/.ez_training 目录的权限处理。
   - load_settings() / save_settings() 的并发安全性。
   - 配置文件格式变更时的向后兼容。
   - _DEFAULT_SETTINGS 中默认值的合理性。

2. **设备检测**
   - detect_devices() 中 torch.cuda 调用在 torch 未安装时的处理。
   - 多 GPU 环境下的设备名称展示。
   - MPS（Apple Silicon）等非 CUDA 设备的支持。

3. **GitHub 镜像**
   - get_github_mirror_prefix() 返回值的格式一致性。
   - 镜像 URL 在 API 请求和文件下载中的正确应用。

4. **工具函数**
   - strip_ansi() 的正则是否覆盖所有 ANSI 转义序列？
   - open_path() 的跨平台实现（Windows/Linux/macOS）。
   - SUPPORTED_IMAGE_FORMATS 是否需要支持更多格式（如 webp、avif）？

5. **代码质量**
   - 常量命名是否符合 Python 规范？
   - 模块间的依赖关系是否清晰？是否有循环导入风险？
   - 类型注解的使用是否充分？
```

---

## 全局 Review 提示词

以下提示词可用于对整个项目进行横向 Review：

### 跨模块一致性

```
请对 EZ Traing 项目进行跨模块一致性审查：

1. **标注格式处理一致性**
   - VOC XML 的读写在 labeling/pascal_voc_io.py、prelabeling/voc_writer.py、
     annotation_scripts/voc_utils.py、data_prep/converter.py 四处实现，
     是否存在行为差异？是否应抽取为统一工具？

2. **YOLO 格式处理一致性**
   - YOLO TXT 的读写在 labeling/yolo_io.py、data_prep/converter.py、
     pages/batch_annotation_page.py 等多处实现，是否一致？

3. **图片路径处理**
   - 中文路径的处理是否在所有 imread/imwrite 调用中统一使用 imread_unicode？
   - 路径分隔符在 Windows 下是否统一处理？

4. **错误处理模式**
   - 各模块的异常处理策略是否统一（静默忽略 vs 弹窗提示 vs 日志记录）？
   - QThread 中的异常是否都通过信号传递到 UI 层？

5. **线程模式**
   - QThread 的使用模式是否统一（继承 vs moveToThread）？
   - 线程取消机制是否统一（标志位检查 vs isInterruptionRequested）？
```

### 性能与资源

```
请对 EZ Traing 项目进行性能和资源使用审查：

1. **内存管理**
   - 大量图片加载时的内存峰值。
   - QPixmap / QImage 缓存是否有上限？
   - OpenCV Mat 对象是否及时释放？

2. **磁盘 IO**
   - 批量文件操作是否有不必要的重复读取？
   - 大文件复制（图片、模型权重）的效率。

3. **GPU 资源**
   - 训练/验证/预标注的 GPU 显存管理。
   - 任务结束后显存是否完全释放？

4. **UI 响应性**
   - 所有耗时操作是否都在后台线程中执行？
   - 信号槽的连接方式（DirectConnection vs QueuedConnection）是否正确？
```

### 安全性

```
请对 EZ Traing 项目进行安全性审查：

1. **敏感信息**
   - API Key 在 vision_api_config.json 中是否明文存储？
   - 日志中是否可能泄露 API Key？

2. **文件操作**
   - 用户提供的路径是否有路径遍历检查？
   - 临时文件的创建和清理。
   - Zip 解压的路径安全性。

3. **外部输入**
   - 用户脚本（script_annotation）的执行安全性。
   - 从 API 返回的 JSON 数据的校验。
   - XML 解析的 XXE 攻击防护。

4. **网络通信**
   - HTTPS 证书验证状态。
   - 请求超时设置。
   - DNS rebinding 防护。
```

---

## 使用说明

1. **单功能 Review**：选择对应功能章节的提示词，将其发送给 AI，同时附上相关源代码文件。
2. **跨模块 Review**：使用"全局 Review 提示词"章节的提示词进行横向审查。
3. **自定义**：可根据实际需要调整提示词中的关注点优先级。
4. **迭代**：首次 Review 发现的问题修复后，可再次使用同一提示词进行回归审查。

> 建议 Review 顺序：先进行全局横向审查，再按功能优先级（数据集管理 → 标注 → 预标注 → 数据准备 → 训练 → 验证）逐一深入。
