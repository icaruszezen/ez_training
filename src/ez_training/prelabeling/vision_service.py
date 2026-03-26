"""视觉模型服务，负责与 OpenAI 兼容 API 进行通信"""

import base64
import json
import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import requests

from ez_training.prelabeling.config import APIConfigManager
from ez_training.prelabeling.models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)



class VisionModelService:
    """视觉模型服务

    负责图片编码、API 请求构建、模型调用和响应解析。
    """

    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0
    _RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    IMAGE_SIZE_WARNING_BYTES = 20 * 1024 * 1024  # 20 MB

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    MIME_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }

    _ENCODE_CACHE_MAX = 16

    def __init__(self, config_manager: APIConfigManager):
        self._config_manager = config_manager
        self._reference_encode_cache: OrderedDict[Tuple[str, int], Tuple[str, str]] = OrderedDict()

    def encode_image_base64(self, image_path: str) -> Tuple[str, str]:
        """将图片文件读取并编码为 base64 格式。

        Args:
            image_path: 图片文件路径

        Returns:
            (base64_data, mime_type) 元组

        Raises:
            ValueError: 图片格式不支持
            FileNotFoundError: 图片文件不存在
            OSError: 图片文件读取失败
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的图片格式: {suffix}，"
                f"支持的格式: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        mime_type = self.MIME_TYPES[suffix]

        file_size = path.stat().st_size
        if file_size > self.IMAGE_SIZE_WARNING_BYTES:
            logger.warning(
                "图片文件较大 (%.1f MB)，编码可能消耗较多内存: %s",
                file_size / (1024 * 1024),
                image_path,
            )

        with open(path, "rb") as f:
            image_data = f.read()

        base64_data = base64.b64encode(image_data).decode("utf-8")
        return base64_data, mime_type

    def encode_reference_images(
        self, image_paths: List[str]
    ) -> List[Tuple[str, str]]:
        """批量编码参考图片为 base64 格式。

        逐张调用 encode_image_base64 进行编码，编码失败的图片会被跳过并记录警告日志。

        Args:
            image_paths: 参考图片文件路径列表

        Returns:
            成功编码的 (base64_data, mime_type) 元组列表
        """
        started_at = perf_counter()
        encoded: List[Tuple[str, str]] = []
        cache_hits = 0
        for path in image_paths:
            try:
                file_path = Path(path).resolve()
                mtime_ns = file_path.stat().st_mtime_ns
                cache_key = (str(file_path), mtime_ns)
                cached = self._reference_encode_cache.get(cache_key)
                if cached is not None:
                    self._reference_encode_cache.move_to_end(cache_key)
                    encoded.append(cached)
                    cache_hits += 1
                    continue

                result = self.encode_image_base64(path)
                encoded.append(result)
                stale_keys = [
                    k for k in self._reference_encode_cache.keys()
                    if k[0] == str(file_path) and k != cache_key
                ]
                for key in stale_keys:
                    self._reference_encode_cache.pop(key, None)
                self._reference_encode_cache[cache_key] = result
                while len(self._reference_encode_cache) > self._ENCODE_CACHE_MAX:
                    self._reference_encode_cache.popitem(last=False)
            except (FileNotFoundError, ValueError, OSError) as e:
                logger.warning("参考图片编码失败，已跳过 %s: %s", path, e)
        logger.info(
            "参考图片编码完成: %d 张，缓存命中 %d，耗时 %.3fs",
            len(encoded),
            cache_hits,
            perf_counter() - started_at,
        )
        return encoded

    def generate_reference_prompt(
        self,
        num_references: int,
        user_description: str = "",
    ) -> str:
        """生成参考图片模式的提示词。

        生成指导视觉模型根据参考图片进行目标检测的提示词，包含：
        1. 参考图片用途说明（需求 5.1）
        2. 在待检测图片中查找相似目标的指令（需求 5.2）
        3. 用户额外描述（如果提供）（需求 5.3）
        4. JSON 格式检测结果要求（需求 5.4）

        Args:
            num_references: 参考图片数量
            user_description: 用户提供的额外文本描述

        Returns:
            完整的提示词字符串
        """
        if num_references == 1:
            ref_desc = "1 张参考图片"
        else:
            ref_desc = f"{num_references} 张参考图片"

        parts: list[str] = []

        # 参考图片用途说明 (需求 5.1)
        parts.append(
            f"我提供了 {ref_desc} 作为目标物体的示例。"
            "这些参考图片展示了我想要检测的目标物体的外观。"
        )

        # 查找相似目标的指令 (需求 5.2)
        parts.append(
            "请在最后一张图片（待检测图片）中找出所有与参考图片中目标物体相同或相似的物体，"
            "并给出每个物体的位置。"
        )

        # 用户额外描述 (需求 5.3)
        desc = user_description.strip() if user_description else ""
        if desc:
            parts.append(f"补充说明：{desc}")

        # JSON 格式要求 (需求 5.4) — 与 parse_response 期望的格式一致
        parts.append(
            "请以 JSON 格式返回检测结果，格式如下：\n"
            '{"objects": [{"label": "物体类别", '
            '"bbox": [x_min, y_min, x_max, y_max], '
            '"confidence": 0.95}]}\n'
            "其中 bbox 为像素坐标，confidence 为置信度（0-1）。"
            "如果没有找到匹配的目标，请返回空数组：{\"objects\": []}"
        )

        return "\n\n".join(parts)

    def build_reference_image_payload(
        self,
        reference_images: List[Tuple[str, str]],  # [(base64, mime_type), ...]
        target_image: Tuple[str, str],  # (base64, mime_type)
        user_description: str = "",
    ) -> dict:
        """构建包含参考图片的 API 请求体。

        按照 OpenAI Vision API 规范组装多图片请求体：
        - 所有参考图片作为 image_url content items
        - 待检测图片作为最后一个 image_url content item
        - 由 generate_reference_prompt 生成的文本提示词

        Args:
            reference_images: 参考图片的 (base64_data, mime_type) 元组列表
            target_image: 待检测图片的 (base64_data, mime_type) 元组
            user_description: 用户提供的额外文本描述

        Returns:
            符合 OpenAI Vision API 规范的请求体字典
        """
        config = self._config_manager.get_config()

        content: list = []

        # 参考图片 image_url items
        for base64_data, mime_type in reference_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}"
                    },
                }
            )

        # 待检测图片 image_url item
        target_base64, target_mime = target_image
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{target_mime};base64,{target_base64}"
                },
            }
        )

        # 文本提示词
        prompt = self.generate_reference_prompt(
            num_references=len(reference_images),
            user_description=user_description,
        )
        content.append({"type": "text", "text": prompt})

        return {
            "model": config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": 4096,
        }

    def build_request_payload(
        self, base64_image: str, mime_type: str, prompt: str
    ) -> dict:
        """构建 OpenAI Vision API 请求体

        按照 OpenAI Vision API 规范组装请求体，包含 model、messages、max_tokens 字段。

        Args:
            base64_image: 图片的 base64 编码字符串
            mime_type: 图片的 MIME 类型（如 image/jpeg）
            prompt: 用户提示词

        Returns:
            符合 OpenAI Vision API 规范的请求体字典
        """
        config = self._config_manager.get_config()
        return {
            "model": config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }
    def _send_request(self, payload: dict) -> DetectionResult:
        """发送 API 请求并解析响应，内置指数退避重试。

        对 ConnectionError、Timeout 和可重试 HTTP 状态码（429/5xx）自动重试，
        最多 MAX_RETRIES 次，每次等待 RETRY_BACKOFF_BASE ** attempt 秒。

        Args:
            payload: 符合 OpenAI Vision API 规范的请求体

        Returns:
            DetectionResult
        """
        config = self._config_manager.get_config()
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        last_error_msg = ""
        last_raw_response = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    config.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=config.timeout,
                )
                response.raise_for_status()
            except requests.Timeout:
                last_error_msg = (
                    f"请求超时（{config.timeout}秒），请检查网络或增加超时时间"
                )
                logger.warning("第 %d/%d 次尝试超时", attempt + 1, self.MAX_RETRIES)
                time.sleep(self.RETRY_BACKOFF_BASE ** attempt)
                continue
            except requests.ConnectionError as e:
                last_error_msg = f"网络连接失败: {e}"
                logger.warning(
                    "第 %d/%d 次尝试连接失败: %s",
                    attempt + 1, self.MAX_RETRIES, e,
                )
                time.sleep(self.RETRY_BACKOFF_BASE ** attempt)
                continue
            except requests.HTTPError as e:
                status_code = response.status_code
                last_raw_response = response.text
                if status_code == 413:
                    msg = (
                        f"API 请求失败 (HTTP {status_code}): 请求体过大，"
                        "请尝试减少参考图片数量或压缩图片"
                    )
                    logger.error(msg)
                    return DetectionResult(
                        success=False, error_message=msg,
                        raw_response=response.text,
                    )
                if status_code in self._RETRYABLE_STATUS_CODES:
                    last_error_msg = (
                        f"API 请求失败 (HTTP {status_code}): {e}"
                    )
                    logger.warning(
                        "第 %d/%d 次尝试收到 HTTP %d，将重试",
                        attempt + 1, self.MAX_RETRIES, status_code,
                    )
                    time.sleep(self.RETRY_BACKOFF_BASE ** attempt)
                    continue
                msg = f"API 请求失败 (HTTP {status_code}): {e}"
                logger.error(msg)
                return DetectionResult(
                    success=False, error_message=msg,
                    raw_response=response.text,
                )
            except requests.RequestException as e:
                msg = f"请求异常: {e}"
                logger.error(msg)
                return DetectionResult(success=False, error_message=msg)

            # 解析响应
            try:
                resp_data = response.json()
                content_text = resp_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as e:
                msg = f"响应结构异常: {e}"
                logger.error("%s\n原始响应: %s", msg, response.text)
                return DetectionResult(
                    success=False, error_message=msg,
                    raw_response=response.text,
                )

            try:
                boxes = self.parse_response(content_text)
            except ValueError as e:
                logger.error("响应解析失败: %s", e)
                return DetectionResult(
                    success=False, error_message=str(e),
                    raw_response=content_text,
                )

            return DetectionResult(
                success=True, boxes=boxes, raw_response=content_text,
            )

        logger.error("重试 %d 次后仍失败: %s", self.MAX_RETRIES, last_error_msg)
        return DetectionResult(
            success=False, error_message=last_error_msg,
            raw_response=last_raw_response,
        )

    def detect_objects(self, image_path: str, prompt: str) -> DetectionResult:
        """调用模型检测图片中的目标

        流程：编码图片 -> 构建请求体 -> 发送 POST 请求 -> 解析响应。

        Args:
            image_path: 图片文件路径
            prompt: 用户提示词

        Returns:
            DetectionResult，成功时 success=True 且 boxes 包含检测结果，
            失败时 success=False 且 error_message 包含错误信息。
        """
        try:
            base64_data, mime_type = self.encode_image_base64(image_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.error("图片编码失败: %s", e)
            return DetectionResult(success=False, error_message=f"图片编码失败: {e}")

        payload = self.build_request_payload(base64_data, mime_type, prompt)
        return self._send_request(payload)

    def detect_objects_with_reference(
        self,
        reference_paths: List[str],
        target_path: str,
        user_description: str = "",
    ) -> DetectionResult:
        """使用参考图片进行目标检测。

        整合参考图片编码、待检测图片编码、请求构建、API 调用和响应解析。

        Args:
            reference_paths: 参考图片文件路径列表
            target_path: 待检测图片文件路径
            user_description: 用户提供的额外文本描述

        Returns:
            DetectionResult，成功时 success=True 且 boxes 包含检测结果，
            失败时 success=False 且 error_message 包含错误信息。
        """
        encoded_references = self.encode_reference_images(reference_paths)
        if not encoded_references:
            msg = "所有参考图片编码失败，请检查参考图片文件是否有效"
            logger.error(msg)
            return DetectionResult(success=False, error_message=msg)

        try:
            target_image = self.encode_image_base64(target_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.error("待检测图片编码失败: %s", e)
            return DetectionResult(
                success=False, error_message=f"待检测图片编码失败: {e}"
            )

        payload = self.build_reference_image_payload(
            reference_images=encoded_references,
            target_image=target_image,
            user_description=user_description,
        )
        return self._send_request(payload)



    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON 字符串。

        LLM 响应可能将 JSON 包裹在 markdown 代码块中，此方法尝试提取纯 JSON。

        Args:
            text: 原始响应文本

        Returns:
            提取出的 JSON 字符串
        """
        stripped = text.strip()

        match = re.search(r"```\w*\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 尝试找到第一个 { 和最后一个 } 之间的内容
        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return stripped[first_brace:last_brace + 1]

        return stripped

    def parse_response(self, response_text: str) -> List[BoundingBox]:
        """解析模型响应，提取边界框信息。

        解析 JSON 响应中的 objects 数组，提取 label、bbox、confidence 信息。
        支持从 markdown 代码块中提取 JSON。

        Args:
            response_text: 模型返回的响应文本

        Returns:
            BoundingBox 列表

        Raises:
            ValueError: 响应格式异常（非 JSON、缺少必要字段、字段类型错误），
                        异常消息中包含原始响应内容
        """
        if not response_text or not response_text.strip():
            raise ValueError("响应为空")

        json_str = self._extract_json(response_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"响应不是有效的 JSON 格式: {e}\n原始响应: {response_text}"
            )

        if not isinstance(data, dict):
            raise ValueError(
                f"响应格式异常: 期望 JSON 对象，得到 {type(data).__name__}\n"
                f"原始响应: {response_text}"
            )

        if "objects" not in data:
            raise ValueError(
                f"响应缺少 'objects' 字段\n原始响应: {response_text}"
            )

        objects = data["objects"]
        if not isinstance(objects, list):
            raise ValueError(
                f"'objects' 字段应为数组，得到 {type(objects).__name__}\n"
                f"原始响应: {response_text}"
            )

        boxes: List[BoundingBox] = []
        for i, obj in enumerate(objects):
            if not isinstance(obj, dict):
                logger.warning("objects[%d] 不是对象，已跳过", i)
                continue

            # 提取 label（必需）
            label = obj.get("label")
            if not isinstance(label, str) or not label.strip():
                logger.warning("objects[%d] 缺少有效的 label 字段，已跳过", i)
                continue

            # 提取 bbox（必需）
            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                logger.warning("objects[%d] 的 bbox 格式无效，已跳过", i)
                continue

            try:
                coords = [int(round(float(v))) for v in bbox]
            except (TypeError, ValueError):
                logger.warning("objects[%d] 的 bbox 坐标值无效，已跳过", i)
                continue

            x_min, y_min, x_max, y_max = coords

            if x_min < 0 or y_min < 0:
                logger.warning(
                    "objects[%d] 包含负坐标 (%d,%d,%d,%d)，已跳过",
                    i, x_min, y_min, x_max, y_max,
                )
                continue

            if x_min >= x_max or y_min >= y_max:
                logger.warning(
                    "objects[%d] 为退化框 (%d,%d,%d,%d)，已跳过",
                    i, x_min, y_min, x_max, y_max,
                )
                continue

            # 提取 confidence（可选，默认 1.0）
            confidence = obj.get("confidence", 1.0)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 1.0

            boxes.append(
                BoundingBox(
                    label=label.strip(),
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=confidence,
                )
            )

        return boxes


