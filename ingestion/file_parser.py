"""
文件解析器模块

支持多种文件格式的解析：TXT、PDF、CSV、Word、Markdown、HTML等
"""

import re
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from core.exceptions import FileParseException

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """文件解析器基类"""

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """
        解析文件并返回文本内容

        :param file_path: 文件路径
        :return: 解析后的文本内容
        """
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """
        检查是否支持该文件类型

        :param file_extension: 文件扩展名（不含点）
        :return: 是否支持
        """
        pass


class TextParser(BaseParser):
    """纯文本文件解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['txt', 'text', 'log', 'md', 'markdown']

    def parse(self, file_path: str) -> str:
        """解析纯文本文件"""
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                # 最后尝试二进制读取并忽略错误
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            logger.info(f"TXT文件解析成功: {file_path}, 长度: {len(content)}")
            return content

        except Exception as e:
            logger.error(f"TXT文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"文本文件解析失败: {str(e)}")


class MarkdownParser(BaseParser):
    """Markdown文件解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['md', 'markdown']

    def parse(self, file_path: str) -> str:
        """解析Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 可选：移除Markdown语法标记，保留纯文本
            # content = self._remove_markdown_syntax(content)

            logger.info(f"Markdown文件解析成功: {file_path}, 长度: {len(content)}")
            return content

        except Exception as e:
            logger.error(f"Markdown文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"Markdown文件解析失败: {str(e)}")

    def _remove_markdown_syntax(self, content: str) -> str:
        """移除Markdown语法（可选）"""
        # 移除标题标记
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # 移除加粗和斜体
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
        content = re.sub(r'\*(.+?)\*', r'\1', content)
        content = re.sub(r'__(.+?)__', r'\1', content)
        content = re.sub(r'_(.+?)_', r'\1', content)
        # 移除链接，保留文本
        content = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', content)
        # 移除图片
        content = re.sub(r'!\[.*?\]\(.+?\)', '', content)
        # 移除代码块
        content = re.sub(r'```[\s\S]*?```', '', content)
        content = re.sub(r'`(.+?)`', r'\1', content)
        return content


class PDFParser(BaseParser):
    """PDF文件解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == 'pdf'

    def parse(self, file_path: str) -> str:
        """解析PDF文件"""
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[第{page_num + 1}页]\n{text}")

            content = "\n\n".join(text_parts)
            logger.info(f"PDF文件解析成功: {file_path}, 页数: {len(reader.pages)}, 长度: {len(content)}")
            return content

        except ImportError:
            logger.warning("pypdf未安装，尝试使用pdfplumber")
            return self._parse_with_pdfplumber(file_path)
        except Exception as e:
            logger.error(f"PDF文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"PDF文件解析失败: {str(e)}")

    def _parse_with_pdfplumber(self, file_path: str) -> str:
        """使用pdfplumber作为备选"""
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"[第{page_num + 1}页]\n{text}")

            content = "\n\n".join(text_parts)
            logger.info(f"PDF文件(pdfplumber)解析成功: {file_path}, 长度: {len(content)}")
            return content

        except ImportError:
            raise FileParseException("PDF解析需要安装 pypdf 或 pdfplumber")
        except Exception as e:
            raise FileParseException(f"PDF文件解析失败: {str(e)}")


class CSVParser(BaseParser):
    """CSV文件解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == 'csv'

    def parse(self, file_path: str) -> str:
        """解析CSV文件"""
        try:
            import csv

            content_parts = []
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        reader = csv.reader(f)
                        rows = list(reader)

                        if rows:
                            # 添加表头信息
                            headers = rows[0] if rows else []
                            content_parts.append("CSV数据表结构:")
                            content_parts.append(" | ".join(headers))
                            content_parts.append("-" * 50)

                            # 添加数据行
                            for row_idx, row in enumerate(rows[1:], start=1):
                                row_text = " | ".join(str(cell) for cell in row)
                                content_parts.append(f"[行{row_idx}] {row_text}")

                    break
                except UnicodeDecodeError:
                    continue

            content = "\n".join(content_parts)
            logger.info(f"CSV文件解析成功: {file_path}, 行数: {len(content_parts)}")
            return content

        except Exception as e:
            logger.error(f"CSV文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"CSV文件解析失败: {str(e)}")


class WordParser(BaseParser):
    """Word文档解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['docx', 'doc']

    def parse(self, file_path: str) -> str:
        """解析Word文档"""
        try:
            from docx import Document

            doc = Document(file_path)
            content_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    content_parts.append(para.text)

            # 添加表格内容
            for table_idx, table in enumerate(doc.tables):
                content_parts.append(f"\n[表格{table_idx + 1}]")
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    content_parts.append(row_text)

            content = "\n\n".join(content_parts)
            logger.info(f"Word文件解析成功: {file_path}, 段落数: {len(doc.paragraphs)}")
            return content

        except ImportError:
            raise FileParseException("Word文档解析需要安装 python-docx")
        except Exception as e:
            logger.error(f"Word文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"Word文件解析失败: {str(e)}")


class ExcelParser(BaseParser):
    """Excel文件解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['xlsx', 'xls']

    def parse(self, file_path: str) -> str:
        """解析Excel文件"""
        try:
            import openpyxl

            wb = openpyxl.load_workbook(file_path, data_only=True)
            content_parts = []

            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                content_parts.append(f"\n=== 工作表: {sheet_name} ===")

                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                    if row_text.strip():
                        content_parts.append(row_text)

            content = "\n".join(content_parts)
            logger.info(f"Excel文件解析成功: {file_path}, 工作表数: {len(wb.sheetnames)}")
            return content

        except ImportError:
            raise FileParseException("Excel文件解析需要安装 openpyxl")
        except Exception as e:
            logger.error(f"Excel文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"Excel文件解析失败: {str(e)}")


class HTMLParser(BaseParser):
    """HTML文件解析器"""

    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['html', 'htm']

    def parse(self, file_path: str) -> str:
        """解析HTML文件"""
        try:
            from bs4 import BeautifulSoup

            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()

            # 获取文本
            content = soup.get_text(separator='\n', strip=True)

            # 清理空行
            lines = [line for line in content.split('\n') if line.strip()]
            content = '\n'.join(lines)

            logger.info(f"HTML文件解析成功: {file_path}, 长度: {len(content)}")
            return content

        except ImportError:
            raise FileParseException("HTML解析需要安装 beautifulsoup4")
        except Exception as e:
            logger.error(f"HTML文件解析失败: {file_path}, 错误: {e}")
            raise FileParseException(f"HTML文件解析失败: {str(e)}")


class FileParserRegistry:
    """文件解析器注册表"""

    def __init__(self):
        self._parsers: List[BaseParser] = []
        self._register_default_parsers()

    def _register_default_parsers(self):
        """注册默认解析器"""
        self._parsers = [
            TextParser(),
            MarkdownParser(),
            PDFParser(),
            CSVParser(),
            WordParser(),
            ExcelParser(),
            HTMLParser(),
        ]
        logger.info(f"已注册 {len(self._parsers)} 个文件解析器")

    def get_parser(self, file_extension: str) -> Optional[BaseParser]:
        """
        获取对应的文件解析器

        :param file_extension: 文件扩展名（不含点）
        :return: 解析器实例，未找到返回None
        """
        for parser in self._parsers:
            if parser.supports(file_extension):
                return parser
        return None

    def parse_file(self, file_path: str) -> str:
        """
        根据文件扩展名自动选择解析器

        :param file_path: 文件路径
        :return: 解析后的文本内容
        """
        file_extension = Path(file_path).suffix.lstrip('.')
        parser = self.get_parser(file_extension)

        if parser is None:
            raise FileParseException(f"不支持的文件类型: {file_extension}")

        return parser.parse(file_path)

    def register_parser(self, parser: BaseParser):
        """注册新的解析器"""
        self._parsers.append(parser)
        logger.info(f"注册新的解析器: {parser.__class__.__name__}")


# 全局解析器实例
_file_parser_registry: Optional[FileParserRegistry] = None


def get_file_parser() -> FileParserRegistry:
    """获取文件解析器注册表单例"""
    global _file_parser_registry
    if _file_parser_registry is None:
        _file_parser_registry = FileParserRegistry()
    return _file_parser_registry