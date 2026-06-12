"""
文本清洗和预处理模块

提供各种文本清洗和规范化功能
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextCleanOptions:
    """文本清洗选项"""
    remove_extra_whitespace: bool = True  # 替换连续空格、换行符和制表符
    remove_urls: bool = False  # 删除所有URL
    remove_emails: bool = False  # 删除所有电子邮件地址
    remove_special_chars: bool = False  # 移除特殊字符
    lowercase: bool = False  # 转换为小写
    remove_numbers: bool = False  # 移除数字
    remove_punctuation: bool = False  # 移除标点符号


class TextCleaner:
    """文本清洗器"""

    # URL正则表达式
    URL_PATTERN = re.compile(
        r'https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    # 电子邮件正则表达式
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        re.IGNORECASE
    )

    # 特殊字符模式（保留中文、英文、数字和基本标点）
    SPECIAL_CHARS_PATTERN = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,;:!?，。；：！？、]')

    def __init__(self, options: Optional[TextCleanOptions] = None):
        self.options = options or TextCleanOptions()

    def clean(self, text: str) -> str:
        """
        清洗文本

        :param text: 原始文本
        :return: 清洗后的文本
        """
        if not text:
            return ""

        cleaned_text = text

        # 移除额外空白
        if self.options.remove_extra_whitespace:
            cleaned_text = self._remove_extra_whitespace(cleaned_text)

        # 移除URL
        if self.options.remove_urls:
            cleaned_text = self._remove_urls(cleaned_text)

        # 移除电子邮件
        if self.options.remove_emails:
            cleaned_text = self._remove_emails(cleaned_text)

        # 转小写
        if self.options.lowercase:
            cleaned_text = cleaned_text.lower()

        # 移除特殊字符
        if self.options.remove_special_chars:
            cleaned_text = self._remove_special_chars(cleaned_text)

        # 移除数字
        if self.options.remove_numbers:
            cleaned_text = self._remove_numbers(cleaned_text)

        # 移除标点
        if self.options.remove_punctuation:
            cleaned_text = self._remove_punctuation(cleaned_text)

        # 清理首尾空白
        cleaned_text = cleaned_text.strip()

        logger.debug(f"文本清洗完成，原长度: {len(text)}, 清洗后长度: {len(cleaned_text)}")
        return cleaned_text

    def _remove_extra_whitespace(self, text: str) -> str:
        """移除额外空白字符"""
        # 替换连续空白为单个空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 替换连续换行为双换行（保留段落结构）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def _remove_urls(self, text: str) -> str:
        """移除URL"""
        # 方法1：使用正则匹配
        text = self.URL_PATTERN.sub('', text)
        # 方法2：通用URL模式
        text = re.sub(r'https?://\S+', '', text)
        return text

    def _remove_emails(self, text: str) -> str:
        """移除电子邮件地址"""
        return self.EMAIL_PATTERN.sub('', text)

    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        return self.SPECIAL_CHARS_PATTERN.sub('', text)

    def _remove_numbers(self, text: str) -> str:
        """移除数字"""
        return re.sub(r'\d+', '', text)

    def _remove_punctuation(self, text: str) -> str:
        """移除标点符号"""
        return re.sub(r'[.,;:!?，。；：！？、""''（）【】《》]', '', text)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗文本"""
        return [self.clean(text) for text in texts]


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self, options: Optional[TextCleanOptions] = None):
        self.cleaner = TextCleaner(options)

    def preprocess(self, text: str, file_type: str = "txt") -> str:
        """
        预处理文本

        :param text: 原始文本
        :param file_type: 文件类型
        :return: 预处理后的文本
        """
        if not text:
            return ""

        # 基础清洗
        text = self.cleaner.clean(text)

        # 根据文件类型进行特定处理
        if file_type.lower() == "csv":
            text = self._preprocess_csv_text(text)
        elif file_type.lower() in ["xlsx", "xls"]:
            text = self._preprocess_excel_text(text)
        elif file_type.lower() in ["pdf"]:
            text = self._preprocess_pdf_text(text)

        return text

    def _preprocess_csv_text(self, text: str) -> str:
        """CSV文本预处理"""
        # CSV可能有表格结构，保持格式
        lines = text.split('\n')
        processed_lines = []

        for line in lines:
            # 清理每行的多余空白
            line = re.sub(r'\s*\|\s*', ' | ', line)
            line = re.sub(r'\s{2,}', ' ', line)
            if line.strip():
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def _preprocess_excel_text(self, text: str) -> str:
        """Excel文本预处理"""
        # 保持表格结构
        return self._preprocess_csv_text(text)

    def _preprocess_pdf_text(self, text: str) -> str:
        """PDF文本预处理"""
        # PDF可能包含页码和奇怪的换行
        # 清理单行结尾的短词（可能是单词被断开）
        lines = text.split('\n')
        processed_lines = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 如果当前行很短且不是段落结束，可能是单词被断开
            if len(line) < 20 and i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                # 合并短行到下一行
                if next_line and not line.endswith(('.', '。', '!', '！', '?', '？')):
                    continue

            processed_lines.append(line)

        return '\n\n'.join(processed_lines)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子

        :param text: 输入文本
        :return: 句子列表
        """
        # 中英文句子分割
        # 匹配常见句末标点
        sentence_endings = r'[.。!！?？;；\n]+'
        sentences = re.split(sentence_endings, text)

        # 清理和过滤
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:  # 过滤过短的片段
                cleaned_sentences.append(sentence)

        return cleaned_sentences


def create_cleaner_from_options(options_dict: Dict[str, Any]) -> TextCleaner:
    """
    从选项字典创建清洗器

    :param options_dict: 选项字典
    :return: TextCleaner实例
    """
    options = TextCleanOptions(
        remove_extra_whitespace=options_dict.get('ddl_option1', True),
        remove_urls=options_dict.get('ddl_option2', False),
        remove_emails=options_dict.get('ddl_option2', False),  # 同一选项
    )
    return TextCleaner(options)