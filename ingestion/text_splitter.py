"""
基于LangChain的智能文本分割器

提供多种文本分割策略，支持自定义分隔符、块大小和重叠
"""

import codecs
import datetime
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from core.exceptions import FileSplitException

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """文本分割配置"""
    chunk_size: int = 1000  # 块大小（字符数或token数）
    chunk_overlap: int = 200  # 块重叠大小
    separators: Optional[List[str]] = None  # 自定义分隔符
    is_separator_regex: bool = False  # 分隔符是否为正则表达式
    length_function: Optional[Callable[[str], int]] = None  # 长度计算函数


class TextSplitterFactory:
    """文本分割器工厂"""

    # 默认分隔符（按优先级排序）
    DEFAULT_SEPARATORS = [
        "\n\n",  # 段落分隔
        "\n",    # 换行
        "。",    # 中文句号
        "！",    # 中文感叹号
        "？",    # 中文问号
        "；",    # 中文分号
        ",",     # 英文逗号
        "，",    # 中文逗号
        ".",     # 英文句号
        " ",     # 空格
        "",      # 字符级分割
    ]

    @staticmethod
    def create_splitter(
        file_type: str,
        config: SplitConfig
    ) -> Any:
        """
        根据文件类型创建合适的分割器

        :param file_type: 文件类型 (txt, md, pdf, csv, python等)
        :param file_type: 文件类型
        :param config: 分割配置
        :return: LangChain文本分割器
        """
        file_type = file_type.lower()
        separators = config.separators or TextSplitterFactory.DEFAULT_SEPARATORS

        try:
            # 统一使用 RecursiveCharacterTextSplitter，传入自定义分隔符
            # MarkdownTextSplitter/PythonTextSplitter 内部已预设 separators，
            # 外部传入会导致 "multiple values for keyword argument 'separators'" 错误
            splitter = RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=config.length_function or len,
                # 关键修复：强制遵守 chunk_size 限制
                keep_separator=False,
                add_start_index=False,
                strip_whitespace=True,
            )

            logger.info(f"创建分割器成功: {file_type}, chunk_size={config.chunk_size}")
            return splitter

        except Exception as e:
            logger.error(f"创建分割器失败: {e}")
            raise FileSplitException(f"创建分割器失败: {str(e)}")

    @staticmethod
    def _get_chunk_params(config: SplitConfig) -> Dict[str, Any]:
        """获取分割器参数"""
        params = {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }
        if config.length_function:
            params["length_function"] = config.length_function
        return params

    @staticmethod
    def parse_separators(separators_str: str) -> List[str]:
        """
        解析分隔符字符串

        支持格式：
        - 逗号分隔: "\n\n,\n,\n, "
        - JSON数组: ["\n\n", "\n", "。"]

        :param separators_str: 分隔符字符串
        :return: 分隔符列表
        """
        if not separators_str:
            return TextSplitterFactory.DEFAULT_SEPARATORS

        # 尝试解析为JSON数组
        if separators_str.strip().startswith('['):
            import json
            try:
                return json.loads(separators_str)
            except json.JSONDecodeError:
                pass

        # 逗号分隔
        separators = [s.strip() for s in separators_str.split(',') if s.strip()]

        # 处理转义序列（如 \n、\t 等），将字面量转义转换为实际控制字符
        decoded_separators = []
        for s in separators:
            try:
                # 使用 unicode_escape 解码转义字符
                decoded = codecs.decode(s, 'unicode_escape')
                decoded_separators.append(decoded)
            except Exception as e:
                logger.warning(f"分隔符解码失败: {s}, 使用原始值. 错误: {e}")
                decoded_separators.append(s)
        
        logger.info(f"分隔符解析: 原始={separators}, 解码后={decoded_separators}")
        return decoded_separators


class IntelligentTextSplitter:
    """智能文本分割器

    提供高级文本分割功能，支持：
    - 多种文件类型自动适配
    - 自定义分隔符
    - 语义感知的分割
    - 元数据保留
    """

    def __init__(self):
        self.factory = TextSplitterFactory()

    def split_text(
        self,
        text: str,
        file_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators_str: str = "\n\n,\n",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        分割文本

        :param text: 输入文本
        :param file_type: 文件类型
        :param chunk_size: 块大小
        :param chunk_overlap: 块重叠
        :param separators_str: 分隔符字符串
        :param metadata: 附加元数据
        :return: Document列表
        """
        if not text:
            logger.warning("输入文本为空")
            return []

        try:
            # 解析分隔符
            separators = self.factory.parse_separators(separators_str)

            # 创建配置
            config = SplitConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )

            # 创建分割器
            splitter = self.factory.create_splitter(file_type, config)

            # 执行分割并转换为Document
            base_metadata = metadata or {}
            base_metadata['file_type'] = file_type

            documents = splitter.create_documents([text], metadatas=[base_metadata])

            # 添加索引信息到元数据
            result = []
            for idx, doc in enumerate(documents):
                doc.metadata['chunk_index'] = idx
                doc.metadata['create_time'] = datetime.datetime.now().isoformat()
                doc.metadata['total_chunks'] = len(documents)
                doc.metadata['chunk_size'] = len(doc.page_content)
                result.append(doc)

            logger.info(f"文本分割完成，生成 {len(result)} 个块")
            return result

        except Exception as e:
            logger.error(f"文本分割失败: {e}")
            raise FileSplitException(f"文本分割失败: {str(e)}")

    def split_documents(
        self,
        documents: List[Document],
        config: SplitConfig
    ) -> List[Document]:
        """
        分割Document列表

        :param documents: 输入Document列表
        :param config: 分割配置
        :return: 分割后的Document列表
        """
        if not documents:
            return []

        try:
            splitter = RecursiveCharacterTextSplitter(
                separators=config.separators or self.factory.DEFAULT_SEPARATORS,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=config.length_function,
            )

            return splitter.split_documents(documents)

        except Exception as e:
            logger.error(f"文档分割失败: {e}")
            raise FileSplitException(f"文档分割失败: {str(e)}")


# 全局分割器实例
_text_splitter: Optional[IntelligentTextSplitter] = None


def get_text_splitter() -> IntelligentTextSplitter:
    """获取文本分割器单例"""
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = IntelligentTextSplitter()
    return _text_splitter


def create_split_config_from_input(
    chunk_size: int,
    chunk_overlap: int,
    separators: str
) -> SplitConfig:
    """
    从输入参数创建分割配置

    :param chunk_size: 分块大小
    :param chunk_overlap: 分块重叠
    :param separators: 分隔符字符串
    :return: SplitConfig实例
    """
    return SplitConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TextSplitterFactory.parse_separators(separators),
    )