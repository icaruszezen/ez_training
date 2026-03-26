"""VisionModelService 单元测试 - 图片编码功能"""

import base64

import pytest

from ez_training.prelabeling.config import APIConfigManager
from ez_training.prelabeling.vision_service import VisionModelService


@pytest.fixture
def config_manager(tmp_path):
    """提供 APIConfigManager 实例"""
    config_dir = tmp_path / ".ez_training"
    return APIConfigManager(config_dir=config_dir)


@pytest.fixture
def service(config_manager):
    """提供 VisionModelService 实例"""
    return VisionModelService(config_manager)


def _create_image(tmp_path, filename: str, content: bytes = b"\x89PNG\r\n\x1a\n") -> str:
    """创建测试图片文件并返回路径"""
    path = tmp_path / filename
    path.write_bytes(content)
    return str(path)


class TestVisionModelServiceInit:
    """初始化测试"""

    def test_has_config_manager(self, service, config_manager):
        assert service._config_manager is config_manager

    def test_supported_formats(self):
        expected = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        assert VisionModelService.SUPPORTED_FORMATS == expected

    def test_mime_types_mapping(self):
        mime = VisionModelService.MIME_TYPES
        assert mime[".jpg"] == "image/jpeg"
        assert mime[".jpeg"] == "image/jpeg"
        assert mime[".png"] == "image/png"
        assert mime[".bmp"] == "image/bmp"
        assert mime[".webp"] == "image/webp"

    def test_all_supported_formats_have_mime_type(self):
        """每种支持的格式都有对应的 MIME 类型"""
        for fmt in VisionModelService.SUPPORTED_FORMATS:
            assert fmt in VisionModelService.MIME_TYPES


class TestEncodeImageBase64:
    """encode_image_base64 测试"""

    def test_encode_png(self, service, tmp_path):
        content = b"\x89PNG\r\n\x1a\nfake_png_data"
        path = _create_image(tmp_path, "test.png", content)
        b64_data, mime = service.encode_image_base64(path)
        assert mime == "image/png"
        assert base64.b64decode(b64_data) == content

    def test_encode_jpg(self, service, tmp_path):
        content = b"\xff\xd8\xff\xe0fake_jpg_data"
        path = _create_image(tmp_path, "test.jpg", content)
        b64_data, mime = service.encode_image_base64(path)
        assert mime == "image/jpeg"
        assert base64.b64decode(b64_data) == content

    def test_encode_jpeg(self, service, tmp_path):
        content = b"\xff\xd8\xff\xe0fake_jpeg_data"
        path = _create_image(tmp_path, "test.jpeg", content)
        b64_data, mime = service.encode_image_base64(path)
        assert mime == "image/jpeg"
        assert base64.b64decode(b64_data) == content

    def test_encode_bmp(self, service, tmp_path):
        content = b"BMfake_bmp_data"
        path = _create_image(tmp_path, "test.bmp", content)
        b64_data, mime = service.encode_image_base64(path)
        assert mime == "image/bmp"
        assert base64.b64decode(b64_data) == content

    def test_encode_webp(self, service, tmp_path):
        content = b"RIFFxxxxWEBPfake_webp_data"
        path = _create_image(tmp_path, "test.webp", content)
        b64_data, mime = service.encode_image_base64(path)
        assert mime == "image/webp"
        assert base64.b64decode(b64_data) == content

    def test_roundtrip_preserves_content(self, service, tmp_path):
        """Property 4: 编码后解码应得到原始内容"""
        content = bytes(range(256))
        path = _create_image(tmp_path, "binary.png", content)
        b64_data, _ = service.encode_image_base64(path)
        decoded = base64.b64decode(b64_data)
        assert decoded == content

    def test_uppercase_extension(self, service, tmp_path):
        """大写扩展名也应支持"""
        content = b"fake_png"
        path = tmp_path / "test.PNG"
        path.write_bytes(content)
        b64_data, mime = service.encode_image_base64(str(path))
        assert mime == "image/png"
        assert base64.b64decode(b64_data) == content

    def test_file_not_found(self, service):
        with pytest.raises(FileNotFoundError, match="图片文件不存在"):
            service.encode_image_base64("/nonexistent/path/image.png")

    def test_unsupported_format(self, service, tmp_path):
        path = _create_image(tmp_path, "test.gif", b"GIF89a")
        with pytest.raises(ValueError, match="不支持的图片格式"):
            service.encode_image_base64(path)

    def test_unsupported_format_tiff(self, service, tmp_path):
        path = _create_image(tmp_path, "test.tiff", b"II*\x00")
        with pytest.raises(ValueError, match="不支持的图片格式"):
            service.encode_image_base64(path)

    def test_empty_file(self, service, tmp_path):
        """空文件也应能编码（返回空 base64）"""
        path = _create_image(tmp_path, "empty.png", b"")
        b64_data, mime = service.encode_image_base64(path)
        assert mime == "image/png"
        assert b64_data == ""
        assert base64.b64decode(b64_data) == b""


class TestStubMethods:
    """确认 stub 方法存在"""

    def test_build_request_payload_exists(self, service):
        assert hasattr(service, "build_request_payload")

    def test_detect_objects_exists(self, service):
        assert hasattr(service, "detect_objects")

    def test_parse_response_exists(self, service):
        assert hasattr(service, "parse_response")


class TestBuildRequestPayload:
    """build_request_payload 测试"""

    def test_returns_dict(self, service):
        result = service.build_request_payload("abc123", "image/png", "detect objects")
        assert isinstance(result, dict)

    def test_model_from_config(self, service, config_manager):
        """model 字段应来自配置管理器"""
        config_manager.update_config(
            model_name="custom-vision-model",
            endpoint="http://test",
            api_key="key",
        )
        result = service.build_request_payload("img", "image/png", "prompt")
        assert result["model"] == "custom-vision-model"

    def test_default_model_name(self, service):
        """默认 model 应为 gpt-4-vision-preview"""
        result = service.build_request_payload("img", "image/png", "prompt")
        assert result["model"] == "gpt-4-vision-preview"

    def test_max_tokens(self, service):
        result = service.build_request_payload("img", "image/png", "prompt")
        assert result["max_tokens"] == 4096

    def test_messages_structure(self, service):
        """messages 应为包含一个 user 角色消息的数组"""
        result = service.build_request_payload("img", "image/png", "prompt")
        messages = result["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_content_has_text_and_image(self, service):
        """content 应包含 text 和 image_url 两个元素"""
        result = service.build_request_payload("img", "image/png", "prompt")
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_text_content(self, service):
        """text 元素应包含提示词"""
        result = service.build_request_payload("img", "image/png", "请检测图中的目标")
        text_item = result["messages"][0]["content"][0]
        assert text_item["text"] == "请检测图中的目标"

    def test_image_url_format(self, service):
        """image_url 应为 data:{mime_type};base64,{base64_image} 格式"""
        result = service.build_request_payload("abc123data", "image/jpeg", "prompt")
        image_item = result["messages"][0]["content"][1]
        url = image_item["image_url"]["url"]
        assert url == "data:image/jpeg;base64,abc123data"

    def test_image_url_with_png(self, service):
        result = service.build_request_payload("pngdata", "image/png", "prompt")
        url = result["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert url.endswith("pngdata")

    def test_image_url_with_webp(self, service):
        result = service.build_request_payload("webpdata", "image/webp", "prompt")
        url = result["messages"][0]["content"][1]["image_url"]["url"]
        assert url == "data:image/webp;base64,webpdata"

    def test_empty_prompt(self, service):
        """空提示词也应正常构建请求体"""
        result = service.build_request_payload("img", "image/png", "")
        assert result["messages"][0]["content"][0]["text"] == ""

    def test_empty_base64(self, service):
        """空 base64 也应正常构建请求体"""
        result = service.build_request_payload("", "image/png", "prompt")
        url = result["messages"][0]["content"][1]["image_url"]["url"]
        assert url == "data:image/png;base64,"

    def test_top_level_keys(self, service):
        """请求体应只包含 model、messages、max_tokens 三个顶层字段"""
        result = service.build_request_payload("img", "image/png", "prompt")
        assert set(result.keys()) == {"model", "messages", "max_tokens"}

import json

from ez_training.prelabeling.models import BoundingBox


class TestParseResponse:
    """parse_response 测试"""

    def test_valid_single_object(self, service):
        """解析包含单个目标的有效响应"""
        response = json.dumps({
            "objects": [
                {"label": "cat", "bbox": [10, 20, 100, 200], "confidence": 0.95}
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "cat"
        assert boxes[0].x_min == 10
        assert boxes[0].y_min == 20
        assert boxes[0].x_max == 100
        assert boxes[0].y_max == 200
        assert boxes[0].confidence == 0.95

    def test_valid_multiple_objects(self, service):
        """解析包含多个目标的有效响应"""
        response = json.dumps({
            "objects": [
                {"label": "cat", "bbox": [10, 20, 100, 200], "confidence": 0.9},
                {"label": "dog", "bbox": [50, 60, 150, 250], "confidence": 0.8},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 2
        assert boxes[0].label == "cat"
        assert boxes[1].label == "dog"

    def test_empty_objects_array(self, service):
        """空 objects 数组应返回空列表"""
        response = json.dumps({"objects": []})
        boxes = service.parse_response(response)
        assert boxes == []

    def test_confidence_default(self, service):
        """缺少 confidence 时默认为 1.0"""
        response = json.dumps({
            "objects": [{"label": "car", "bbox": [0, 0, 50, 50]}]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].confidence == 1.0

    def test_float_bbox_rounded(self, service):
        """浮点数坐标应四舍五入为整数"""
        response = json.dumps({
            "objects": [{"label": "car", "bbox": [10.4, 20.6, 100.5, 200.3]}]
        })
        boxes = service.parse_response(response)
        assert boxes[0].x_min == 10
        assert boxes[0].y_min == 21
        assert boxes[0].x_max == 100  # round(100.5) = 100 in Python (banker's rounding)
        assert boxes[0].y_max == 200

    def test_label_stripped(self, service):
        """label 前后空白应被去除"""
        response = json.dumps({
            "objects": [{"label": "  cat  ", "bbox": [0, 0, 10, 10]}]
        })
        boxes = service.parse_response(response)
        assert boxes[0].label == "cat"

    # --- JSON 提取测试 ---

    def test_json_in_markdown_code_block(self, service):
        """从 markdown ```json 代码块中提取 JSON"""
        response = '这是模型的回复：\n```json\n{"objects": [{"label": "cat", "bbox": [1, 2, 3, 4]}]}\n```\n以上是结果。'
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "cat"

    def test_json_in_plain_code_block(self, service):
        """从 markdown ``` 代码块中提取 JSON"""
        response = '```\n{"objects": [{"label": "dog", "bbox": [5, 6, 7, 8]}]}\n```'
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "dog"

    def test_json_with_surrounding_text(self, service):
        """JSON 前后有其他文本"""
        response = 'Here are the results: {"objects": [{"label": "bird", "bbox": [0, 0, 10, 10]}]} end.'
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "bird"

    def test_pure_json(self, service):
        """纯 JSON 字符串"""
        response = '{"objects": [{"label": "fish", "bbox": [1, 2, 3, 4], "confidence": 0.5}]}'
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "fish"
        assert boxes[0].confidence == 0.5

    # --- 异常格式处理测试 ---

    def test_empty_string_raises(self, service):
        """空字符串应抛出 ValueError"""
        with pytest.raises(ValueError, match="响应为空"):
            service.parse_response("")

    def test_whitespace_only_raises(self, service):
        """仅空白字符应抛出 ValueError"""
        with pytest.raises(ValueError, match="响应为空"):
            service.parse_response("   \n\t  ")

    def test_none_raises(self, service):
        """None 应抛出 ValueError"""
        with pytest.raises(ValueError, match="响应为空"):
            service.parse_response(None)

    def test_invalid_json_raises(self, service):
        """无效 JSON 应抛出 ValueError 并包含原始响应"""
        with pytest.raises(ValueError, match="原始响应"):
            service.parse_response("this is not json at all")

    def test_json_array_raises(self, service):
        """JSON 数组（非对象）应抛出 ValueError"""
        with pytest.raises(ValueError, match="期望 JSON 对象"):
            service.parse_response('[1, 2, 3]')

    def test_missing_objects_key_raises(self, service):
        """缺少 objects 字段应抛出 ValueError"""
        with pytest.raises(ValueError, match="缺少 'objects' 字段"):
            service.parse_response('{"results": []}')

    def test_objects_not_array_raises(self, service):
        """objects 不是数组应抛出 ValueError"""
        with pytest.raises(ValueError, match="应为数组"):
            service.parse_response('{"objects": "not an array"}')

    # --- 单个对象字段异常（跳过而非报错）---

    def test_skip_object_without_label(self, service):
        """缺少 label 的对象应被跳过"""
        response = json.dumps({
            "objects": [
                {"bbox": [0, 0, 10, 10]},
                {"label": "valid", "bbox": [1, 2, 3, 4]},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "valid"

    def test_skip_object_with_empty_label(self, service):
        """空 label 的对象应被跳过"""
        response = json.dumps({
            "objects": [
                {"label": "", "bbox": [0, 0, 10, 10]},
                {"label": "ok", "bbox": [1, 2, 3, 4]},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "ok"

    def test_skip_object_without_bbox(self, service):
        """缺少 bbox 的对象应被跳过"""
        response = json.dumps({
            "objects": [
                {"label": "cat"},
                {"label": "dog", "bbox": [1, 2, 3, 4]},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "dog"

    def test_skip_object_with_wrong_bbox_length(self, service):
        """bbox 长度不为 4 的对象应被跳过"""
        response = json.dumps({
            "objects": [
                {"label": "cat", "bbox": [1, 2, 3]},
                {"label": "dog", "bbox": [1, 2, 3, 4]},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "dog"

    def test_skip_object_with_non_numeric_bbox(self, service):
        """bbox 包含非数值的对象应被跳过"""
        response = json.dumps({
            "objects": [
                {"label": "cat", "bbox": ["a", "b", "c", "d"]},
                {"label": "dog", "bbox": [1, 2, 3, 4]},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "dog"

    def test_skip_non_dict_object(self, service):
        """objects 数组中的非字典元素应被跳过"""
        response = json.dumps({
            "objects": [
                "not a dict",
                {"label": "cat", "bbox": [1, 2, 3, 4]},
            ]
        })
        boxes = service.parse_response(response)
        assert len(boxes) == 1
        assert boxes[0].label == "cat"

    def test_invalid_confidence_uses_default(self, service):
        """无效的 confidence 值应使用默认值 1.0"""
        response = json.dumps({
            "objects": [
                {"label": "cat", "bbox": [1, 2, 3, 4], "confidence": "high"}
            ]
        })
        boxes = service.parse_response(response)
        assert boxes[0].confidence == 1.0

    def test_returns_bounding_box_instances(self, service):
        """返回的应是 BoundingBox 实例"""
        response = json.dumps({
            "objects": [{"label": "cat", "bbox": [1, 2, 3, 4]}]
        })
        boxes = service.parse_response(response)
        assert isinstance(boxes[0], BoundingBox)


class TestExtractJson:
    """_extract_json 辅助方法测试"""

    def test_plain_json(self, service):
        text = '{"key": "value"}'
        assert service._extract_json(text) == '{"key": "value"}'

    def test_json_code_block(self, service):
        text = '```json\n{"key": "value"}\n```'
        assert service._extract_json(text) == '{"key": "value"}'

    def test_plain_code_block(self, service):
        text = '```\n{"key": "value"}\n```'
        assert service._extract_json(text) == '{"key": "value"}'

    def test_json_with_prefix_text(self, service):
        text = 'Here is the result: {"key": "value"}'
        assert service._extract_json(text) == '{"key": "value"}'

    def test_json_with_surrounding_text(self, service):
        text = 'prefix {"key": "value"} suffix'
        assert service._extract_json(text) == '{"key": "value"}'

    def test_no_json_returns_stripped(self, service):
        text = '  no json here  '
        assert service._extract_json(text) == 'no json here'


from unittest.mock import MagicMock, patch

import requests

from ez_training.prelabeling.models import DetectionResult


def _setup_service_with_config(tmp_path, endpoint="https://api.example.com/v1/chat/completions",
                                api_key="sk-test-key-123", timeout=30):
    """创建配置好的 service 实例并返回 (service, image_path)"""
    config_dir = tmp_path / ".ez_training"
    mgr = APIConfigManager(config_dir=config_dir)
    mgr.update_config(endpoint=endpoint, api_key=api_key, timeout=timeout)
    svc = VisionModelService(mgr)
    # 创建测试图片
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_data")
    return svc, str(img)


def _make_api_response(content_text, status_code=200):
    """构造模拟的 requests.Response"""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = json.dumps({
        "choices": [{"message": {"content": content_text}}]
    })
    resp.json.return_value = {
        "choices": [{"message": {"content": content_text}}]
    }
    resp.raise_for_status.return_value = None
    return resp


class TestDetectObjects:
    """detect_objects 方法测试"""

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_successful_detection(self, mock_post, tmp_path):
        """成功检测返回 DetectionResult(success=True)"""
        svc, img = _setup_service_with_config(tmp_path)
        content = json.dumps({
            "objects": [{"label": "cat", "bbox": [10, 20, 100, 200], "confidence": 0.9}]
        })
        mock_post.return_value = _make_api_response(content)

        result = svc.detect_objects(img, "detect cats")

        assert result.success is True
        assert len(result.boxes) == 1
        assert result.boxes[0].label == "cat"
        assert result.boxes[0].x_min == 10
        assert result.boxes[0].confidence == 0.9

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_sends_post_to_endpoint(self, mock_post, tmp_path):
        """应向配置的 endpoint 发送 POST 请求"""
        svc, img = _setup_service_with_config(
            tmp_path, endpoint="https://my-api.com/v1/chat/completions"
        )
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects(img, "prompt")

        call_args = mock_post.call_args
        assert call_args[0][0] == "https://my-api.com/v1/chat/completions"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_authorization_header(self, mock_post, tmp_path):
        """请求头应包含 Authorization: Bearer {api_key}"""
        svc, img = _setup_service_with_config(tmp_path, api_key="sk-my-secret")
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects(img, "prompt")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-my-secret"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_content_type_header(self, mock_post, tmp_path):
        """请求头应包含 Content-Type: application/json"""
        svc, img = _setup_service_with_config(tmp_path)
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects(img, "prompt")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_timeout_from_config(self, mock_post, tmp_path):
        """请求超时应使用配置中的 timeout 值"""
        svc, img = _setup_service_with_config(tmp_path, timeout=45)
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects(img, "prompt")

        assert mock_post.call_args[1]["timeout"] == 45

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_request_payload_sent_as_json(self, mock_post, tmp_path):
        """请求体应作为 JSON 发送，包含 model 和 messages"""
        svc, img = _setup_service_with_config(tmp_path)
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects(img, "detect objects")

        payload = mock_post.call_args[1]["json"]
        assert "model" in payload
        assert "messages" in payload
        assert "max_tokens" in payload

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_timeout_error(self, mock_post, tmp_path):
        """请求超时应返回 success=False 并包含超时信息"""
        svc, img = _setup_service_with_config(tmp_path, timeout=10)
        mock_post.side_effect = requests.Timeout("Connection timed out")

        result = svc.detect_objects(img, "prompt")

        assert result.success is False
        assert "超时" in result.error_message
        assert "10" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_connection_error(self, mock_post, tmp_path):
        """连接失败应返回 success=False 并包含网络错误信息"""
        svc, img = _setup_service_with_config(tmp_path)
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        result = svc.detect_objects(img, "prompt")

        assert result.success is False
        assert "网络连接失败" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_http_error(self, mock_post, tmp_path):
        """HTTP 错误应返回 success=False 并包含状态码"""
        svc, img = _setup_service_with_config(tmp_path)
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 401
        resp.text = '{"error": "Unauthorized"}'
        resp.raise_for_status.side_effect = requests.HTTPError(
            "401 Client Error", response=resp
        )
        mock_post.return_value = resp

        result = svc.detect_objects(img, "prompt")

        assert result.success is False
        assert "401" in result.error_message
        assert result.raw_response == '{"error": "Unauthorized"}'

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_invalid_response_structure(self, mock_post, tmp_path):
        """响应缺少 choices 字段应返回 success=False"""
        svc, img = _setup_service_with_config(tmp_path)
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = '{"result": "no choices"}'
        resp.json.return_value = {"result": "no choices"}
        resp.raise_for_status.return_value = None
        mock_post.return_value = resp

        result = svc.detect_objects(img, "prompt")

        assert result.success is False
        assert "响应结构异常" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_parse_response_value_error(self, mock_post, tmp_path):
        """parse_response 抛出 ValueError 应返回 success=False"""
        svc, img = _setup_service_with_config(tmp_path)
        # 返回的 content 不是有效的检测结果 JSON
        content = "this is not valid detection json"
        mock_post.return_value = _make_api_response(content)

        result = svc.detect_objects(img, "prompt")

        assert result.success is False
        assert result.raw_response == content

    def test_image_not_found(self, tmp_path):
        """图片不存在应返回 success=False"""
        config_dir = tmp_path / ".ez_training"
        mgr = APIConfigManager(config_dir=config_dir)
        svc = VisionModelService(mgr)

        result = svc.detect_objects("/nonexistent/image.png", "prompt")

        assert result.success is False
        assert "图片编码失败" in result.error_message

    def test_unsupported_image_format(self, tmp_path):
        """不支持的图片格式应返回 success=False"""
        config_dir = tmp_path / ".ez_training"
        mgr = APIConfigManager(config_dir=config_dir)
        svc = VisionModelService(mgr)
        img = tmp_path / "test.gif"
        img.write_bytes(b"GIF89a")

        result = svc.detect_objects(str(img), "prompt")

        assert result.success is False
        assert "图片编码失败" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_empty_detection_result(self, mock_post, tmp_path):
        """模型返回空 objects 应返回 success=True 且 boxes 为空"""
        svc, img = _setup_service_with_config(tmp_path)
        mock_post.return_value = _make_api_response('{"objects": []}')

        result = svc.detect_objects(img, "prompt")

        assert result.success is True
        assert result.boxes == []

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_multiple_boxes_detected(self, mock_post, tmp_path):
        """多个目标检测结果应全部返回"""
        svc, img = _setup_service_with_config(tmp_path)
        content = json.dumps({
            "objects": [
                {"label": "cat", "bbox": [10, 20, 100, 200]},
                {"label": "dog", "bbox": [50, 60, 150, 250], "confidence": 0.8},
            ]
        })
        mock_post.return_value = _make_api_response(content)

        result = svc.detect_objects(img, "prompt")

        assert result.success is True
        assert len(result.boxes) == 2
        assert result.boxes[0].label == "cat"
        assert result.boxes[1].label == "dog"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_raw_response_on_success(self, mock_post, tmp_path):
        """成功时 raw_response 应包含模型返回的文本内容"""
        svc, img = _setup_service_with_config(tmp_path)
        content = '{"objects": [{"label": "car", "bbox": [0, 0, 50, 50]}]}'
        mock_post.return_value = _make_api_response(content)

        result = svc.detect_objects(img, "prompt")

        assert result.raw_response == content

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_general_request_exception(self, mock_post, tmp_path):
        """其他 requests 异常应返回 success=False"""
        svc, img = _setup_service_with_config(tmp_path)
        mock_post.side_effect = requests.RequestException("Something went wrong")

        result = svc.detect_objects(img, "prompt")

        assert result.success is False
        assert "请求异常" in result.error_message


class TestEncodeReferenceImages:
    """encode_reference_images 批量编码参考图片测试"""

    def test_encode_multiple_valid_images(self, service, tmp_path):
        """多张有效图片应全部编码成功"""
        img1 = _create_image(tmp_path, "ref1.png")
        img2 = _create_image(tmp_path, "ref2.jpg", b"\xff\xd8\xff\xe0")

        result = service.encode_reference_images([img1, img2])

        assert len(result) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
        assert result[0][1] == "image/png"
        assert result[1][1] == "image/jpeg"

    def test_encode_single_image(self, service, tmp_path):
        """单张图片也应正常返回"""
        img = _create_image(tmp_path, "ref.png")

        result = service.encode_reference_images([img])

        assert len(result) == 1

    def test_empty_list_returns_empty(self, service):
        """空列表应返回空列表"""
        result = service.encode_reference_images([])

        assert result == []

    def test_skip_nonexistent_file(self, service, tmp_path):
        """不存在的文件应被跳过"""
        valid = _create_image(tmp_path, "valid.png")
        missing = str(tmp_path / "missing.png")

        result = service.encode_reference_images([valid, missing])

        assert len(result) == 1

    def test_skip_unsupported_format(self, service, tmp_path):
        """不支持的格式应被跳过"""
        valid = _create_image(tmp_path, "valid.png")
        tiff = _create_image(tmp_path, "bad.tiff")

        result = service.encode_reference_images([valid, tiff])

        assert len(result) == 1

    def test_all_invalid_returns_empty(self, service, tmp_path):
        """全部编码失败时应返回空列表"""
        missing = str(tmp_path / "missing.png")
        tiff = _create_image(tmp_path, "bad.tiff")

        result = service.encode_reference_images([missing, tiff])

        assert result == []

    def test_base64_data_is_valid(self, service, tmp_path):
        """返回的 base64 数据应可解码"""
        content = b"test image data"
        img = _create_image(tmp_path, "test.png", content)

        result = service.encode_reference_images([img])

        decoded = base64.b64decode(result[0][0])
        assert decoded == content

    def test_preserves_order(self, service, tmp_path):
        """编码结果应保持输入顺序"""
        img_png = _create_image(tmp_path, "a.png", b"\x89PNG")
        img_jpg = _create_image(tmp_path, "b.jpg", b"\xff\xd8")
        img_webp = _create_image(tmp_path, "c.webp", b"RIFF")

        result = service.encode_reference_images([img_png, img_jpg, img_webp])

        assert result[0][1] == "image/png"
        assert result[1][1] == "image/jpeg"
        assert result[2][1] == "image/webp"

    def test_logs_warning_on_failure(self, service, tmp_path, caplog):
        """编码失败时应记录警告日志"""
        import logging

        missing = str(tmp_path / "missing.png")

        with caplog.at_level(logging.WARNING):
            service.encode_reference_images([missing])

        assert any("参考图片编码失败" in msg for msg in caplog.messages)


class TestGenerateReferencePrompt:
    """generate_reference_prompt 方法测试"""

    def test_contains_reference_purpose(self, service):
        """提示词应包含参考图片用途说明 (需求 5.1)"""
        prompt = service.generate_reference_prompt(num_references=3)
        assert "参考图片" in prompt
        assert "目标物体" in prompt
        assert "示例" in prompt

    def test_contains_find_instruction(self, service):
        """提示词应包含查找相似目标的指令 (需求 5.2)"""
        prompt = service.generate_reference_prompt(num_references=2)
        assert "待检测图片" in prompt
        assert "相似" in prompt or "相同" in prompt

    def test_contains_json_format(self, service):
        """提示词应包含 JSON 格式要求 (需求 5.4)"""
        prompt = service.generate_reference_prompt(num_references=1)
        assert "JSON" in prompt
        assert "objects" in prompt
        assert "label" in prompt
        assert "bbox" in prompt
        assert "confidence" in prompt

    def test_json_format_matches_parse_response(self, service):
        """JSON 格式要求应与 parse_response 期望的格式一致"""
        prompt = service.generate_reference_prompt(num_references=1)
        assert "x_min" in prompt
        assert "y_min" in prompt
        assert "x_max" in prompt
        assert "y_max" in prompt

    def test_includes_user_description(self, service):
        """提供用户描述时应整合到提示词中 (需求 5.3)"""
        prompt = service.generate_reference_prompt(
            num_references=2, user_description="红色的汽车"
        )
        assert "红色的汽车" in prompt

    def test_no_user_description_by_default(self, service):
        """不提供用户描述时提示词不应包含补充说明标记"""
        prompt = service.generate_reference_prompt(num_references=1)
        assert "补充说明" not in prompt

    def test_empty_user_description_ignored(self, service):
        """空字符串用户描述应被忽略"""
        prompt = service.generate_reference_prompt(
            num_references=1, user_description=""
        )
        assert "补充说明" not in prompt

    def test_whitespace_user_description_ignored(self, service):
        """仅空白的用户描述应被忽略"""
        prompt = service.generate_reference_prompt(
            num_references=1, user_description="   \t\n  "
        )
        assert "补充说明" not in prompt

    def test_single_reference_count(self, service):
        """单张参考图片时应正确描述数量"""
        prompt = service.generate_reference_prompt(num_references=1)
        assert "1 张参考图片" in prompt

    def test_multiple_reference_count(self, service):
        """多张参考图片时应正确描述数量"""
        prompt = service.generate_reference_prompt(num_references=5)
        assert "5 张参考图片" in prompt

    def test_returns_string(self, service):
        """应返回字符串类型"""
        result = service.generate_reference_prompt(num_references=1)
        assert isinstance(result, str)

    def test_prompt_not_empty(self, service):
        """生成的提示词不应为空"""
        result = service.generate_reference_prompt(num_references=1)
        assert len(result.strip()) > 0

    def test_empty_objects_instruction(self, service):
        """提示词应包含空结果的返回格式说明"""
        prompt = service.generate_reference_prompt(num_references=1)
        assert '"objects": []' in prompt


class TestBuildReferenceImagePayload:
    """build_reference_image_payload 方法测试"""

    def test_returns_dict(self, service):
        """应返回字典类型"""
        ref_images = [("aGVsbG8=", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        assert isinstance(result, dict)

    def test_top_level_keys(self, service):
        """返回的字典应包含 model、messages、max_tokens"""
        ref_images = [("aGVsbG8=", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        assert set(result.keys()) == {"model", "messages", "max_tokens"}

    def test_model_from_config(self, service, config_manager):
        """model 字段应来自配置"""
        config_manager.update_config(model_name="gpt-4o")
        ref_images = [("aGVsbG8=", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        assert result["model"] == "gpt-4o"

    def test_max_tokens(self, service):
        """max_tokens 应为 4096"""
        ref_images = [("aGVsbG8=", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        assert result["max_tokens"] == 4096

    def test_messages_structure(self, service):
        """messages 应包含一条 user 角色消息"""
        ref_images = [("aGVsbG8=", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)

    def test_single_reference_image_content_count(self, service):
        """1 张参考图片时 content 应有 3 项（1 ref + 1 target + 1 text）"""
        ref_images = [("cmVm", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        assert len(content) == 3

    def test_multiple_reference_images_content_count(self, service):
        """多张参考图片时 content 项数应正确"""
        ref_images = [
            ("cmVmMQ==", "image/png"),
            ("cmVmMg==", "image/jpeg"),
            ("cmVmMw==", "image/webp"),
        ]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        # 3 ref images + 1 target + 1 text = 5
        assert len(content) == 5

    def test_reference_images_come_first(self, service):
        """参考图片应排在 content 列表最前面"""
        ref_images = [
            ("cmVmMQ==", "image/png"),
            ("cmVmMg==", "image/jpeg"),
        ]
        target = ("dGFyZ2V0", "image/webp")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        # First two items should be reference images
        assert content[0]["type"] == "image_url"
        assert "image/png" in content[0]["image_url"]["url"]
        assert content[1]["type"] == "image_url"
        assert "image/jpeg" in content[1]["image_url"]["url"]

    def test_target_image_after_references(self, service):
        """待检测图片应在参考图片之后"""
        ref_images = [("cmVm", "image/png")]
        target = ("dGFyZ2V0", "image/webp")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        # Index 1 should be the target image (after 1 ref image)
        target_item = content[1]
        assert target_item["type"] == "image_url"
        assert "image/webp" in target_item["image_url"]["url"]
        assert "dGFyZ2V0" in target_item["image_url"]["url"]

    def test_text_prompt_is_last(self, service):
        """文本提示词应是 content 列表的最后一项"""
        ref_images = [("cmVm", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        last_item = content[-1]
        assert last_item["type"] == "text"
        assert isinstance(last_item["text"], str)
        assert len(last_item["text"]) > 0

    def test_image_url_format(self, service):
        """image_url 应使用 data URI 格式"""
        ref_images = [("aGVsbG8=", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        ref_url = content[0]["image_url"]["url"]
        assert ref_url == "data:image/png;base64,aGVsbG8="
        target_url = content[1]["image_url"]["url"]
        assert target_url == "data:image/jpeg;base64,dGFyZ2V0"

    def test_prompt_uses_generate_reference_prompt(self, service):
        """文本提示词应由 generate_reference_prompt 生成"""
        ref_images = [("cmVm", "image/png"), ("cmVmMg==", "image/jpeg")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        text = content[-1]["text"]
        expected = service.generate_reference_prompt(num_references=2)
        assert text == expected

    def test_user_description_passed_to_prompt(self, service):
        """user_description 应传递给 generate_reference_prompt"""
        ref_images = [("cmVm", "image/png")]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(
            ref_images, target, user_description="红色的汽车"
        )
        content = result["messages"][0]["content"]
        text = content[-1]["text"]
        assert "红色的汽车" in text

    def test_empty_reference_list(self, service):
        """空参考图片列表时 content 应只有 target + text"""
        ref_images = []
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "text"

    def test_ten_reference_images(self, service):
        """10 张参考图片时应全部包含在请求体中 (需求 4.4)"""
        ref_images = [(f"cmVme{i}", "image/png") for i in range(10)]
        target = ("dGFyZ2V0", "image/jpeg")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        # 10 ref + 1 target + 1 text = 12
        assert len(content) == 12
        image_items = [c for c in content if c["type"] == "image_url"]
        assert len(image_items) == 11  # 10 ref + 1 target

    def test_preserves_reference_image_order(self, service):
        """参考图片应保持输入顺序"""
        ref_images = [
            ("first", "image/png"),
            ("second", "image/jpeg"),
            ("third", "image/webp"),
        ]
        target = ("target", "image/bmp")
        result = service.build_reference_image_payload(ref_images, target)
        content = result["messages"][0]["content"]
        assert "first" in content[0]["image_url"]["url"]
        assert "second" in content[1]["image_url"]["url"]
        assert "third" in content[2]["image_url"]["url"]


class TestDetectObjectsWithReference:
    """detect_objects_with_reference 方法测试"""

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_successful_detection(self, mock_post, tmp_path):
        """成功检测返回 DetectionResult(success=True)"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref1 = _create_image(tmp_path, "ref1.png", b"\x89PNGref1")
        ref2 = _create_image(tmp_path, "ref2.jpg", b"\xff\xd8\xff\xe0ref2")
        target = _create_image(tmp_path, "target.png", b"\x89PNGtarget")

        content = json.dumps({
            "objects": [{"label": "cat", "bbox": [10, 20, 100, 200], "confidence": 0.9}]
        })
        mock_post.return_value = _make_api_response(content)

        result = svc.detect_objects_with_reference([ref1, ref2], target)

        assert result.success is True
        assert len(result.boxes) == 1
        assert result.boxes[0].label == "cat"
        assert result.boxes[0].confidence == 0.9

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_sends_post_to_endpoint(self, mock_post, tmp_path):
        """应向配置的 endpoint 发送 POST 请求"""
        svc, _ = _setup_service_with_config(
            tmp_path, endpoint="https://my-api.com/v1/chat/completions"
        )
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects_with_reference([ref], target)

        assert mock_post.call_args[0][0] == "https://my-api.com/v1/chat/completions"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_authorization_header(self, mock_post, tmp_path):
        """请求头应包含正确的 Authorization"""
        svc, _ = _setup_service_with_config(tmp_path, api_key="sk-ref-key")
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects_with_reference([ref], target)

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-ref-key"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_payload_contains_reference_and_target_images(self, mock_post, tmp_path):
        """请求体应包含参考图片和待检测图片"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref1 = _create_image(tmp_path, "ref1.png", b"\x89PNGref1")
        ref2 = _create_image(tmp_path, "ref2.png", b"\x89PNGref2")
        target = _create_image(tmp_path, "target.png", b"\x89PNGtarget")
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects_with_reference([ref1, ref2], target)

        payload = mock_post.call_args[1]["json"]
        content = payload["messages"][0]["content"]
        image_items = [c for c in content if c["type"] == "image_url"]
        # 2 ref + 1 target = 3 images
        assert len(image_items) == 3

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_user_description_in_payload(self, mock_post, tmp_path):
        """user_description 应包含在请求体的提示词中"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.return_value = _make_api_response('{"objects": []}')

        svc.detect_objects_with_reference([ref], target, user_description="红色的汽车")

        payload = mock_post.call_args[1]["json"]
        content = payload["messages"][0]["content"]
        text_item = [c for c in content if c["type"] == "text"][0]
        assert "红色的汽车" in text_item["text"]

    def test_all_reference_images_invalid(self, tmp_path):
        """所有参考图片编码失败时应返回 success=False (需求 7.2)"""
        svc, _ = _setup_service_with_config(tmp_path)
        missing1 = str(tmp_path / "missing1.png")
        missing2 = str(tmp_path / "missing2.png")
        target = _create_image(tmp_path, "target.png")

        result = svc.detect_objects_with_reference([missing1, missing2], target)

        assert result.success is False
        assert "参考图片编码失败" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_partial_reference_images_valid(self, mock_post, tmp_path):
        """部分参考图片无效时应继续使用有效的图片"""
        svc, _ = _setup_service_with_config(tmp_path)
        valid_ref = _create_image(tmp_path, "valid.png")
        missing_ref = str(tmp_path / "missing.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.return_value = _make_api_response('{"objects": []}')

        result = svc.detect_objects_with_reference([valid_ref, missing_ref], target)

        assert result.success is True
        # Payload should contain only 1 ref image + 1 target
        payload = mock_post.call_args[1]["json"]
        content = payload["messages"][0]["content"]
        image_items = [c for c in content if c["type"] == "image_url"]
        assert len(image_items) == 2

    def test_target_image_not_found(self, tmp_path):
        """待检测图片不存在应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")

        result = svc.detect_objects_with_reference([ref], "/nonexistent/target.png")

        assert result.success is False
        assert "待检测图片编码失败" in result.error_message

    def test_target_image_unsupported_format(self, tmp_path):
        """待检测图片格式不支持应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.gif", b"GIF89a")

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "待检测图片编码失败" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_timeout_error(self, mock_post, tmp_path):
        """请求超时应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path, timeout=15)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.side_effect = requests.Timeout("Connection timed out")

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "超时" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_connection_error(self, mock_post, tmp_path):
        """连接失败应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "网络连接失败" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_http_413_payload_too_large(self, mock_post, tmp_path):
        """HTTP 413 应返回建议减少参考图片的提示 (需求 7.3)"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 413
        resp.text = '{"error": "Payload Too Large"}'
        resp.raise_for_status.side_effect = requests.HTTPError(
            "413 Client Error", response=resp
        )
        mock_post.return_value = resp

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "请求体过大" in result.error_message
        assert "减少参考图片" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_http_error_non_413(self, mock_post, tmp_path):
        """非 413 HTTP 错误应返回标准错误信息"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 500
        resp.text = '{"error": "Internal Server Error"}'
        resp.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error", response=resp
        )
        mock_post.return_value = resp

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "500" in result.error_message
        assert result.raw_response == '{"error": "Internal Server Error"}'

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_invalid_response_structure(self, mock_post, tmp_path):
        """响应缺少 choices 字段应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.text = '{"result": "no choices"}'
        resp.json.return_value = {"result": "no choices"}
        resp.raise_for_status.return_value = None
        mock_post.return_value = resp

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "响应结构异常" in result.error_message

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_parse_response_error(self, mock_post, tmp_path):
        """parse_response 失败应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.return_value = _make_api_response("not valid detection json")

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert result.raw_response == "not valid detection json"

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_empty_detection_result(self, mock_post, tmp_path):
        """模型返回空 objects 应返回 success=True 且 boxes 为空"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.return_value = _make_api_response('{"objects": []}')

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is True
        assert result.boxes == []

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_raw_response_on_success(self, mock_post, tmp_path):
        """成功时 raw_response 应包含模型返回的文本"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        content = '{"objects": [{"label": "car", "bbox": [0, 0, 50, 50]}]}'
        mock_post.return_value = _make_api_response(content)

        result = svc.detect_objects_with_reference([ref], target)

        assert result.raw_response == content

    @patch("ez_training.prelabeling.vision_service.requests.post")
    def test_general_request_exception(self, mock_post, tmp_path):
        """其他 requests 异常应返回 success=False"""
        svc, _ = _setup_service_with_config(tmp_path)
        ref = _create_image(tmp_path, "ref.png")
        target = _create_image(tmp_path, "target.png")
        mock_post.side_effect = requests.RequestException("Something went wrong")

        result = svc.detect_objects_with_reference([ref], target)

        assert result.success is False
        assert "请求异常" in result.error_message
