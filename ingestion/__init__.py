"""
文档摄入管道包

负责源文档（PDF、HTML、TXT 等）的加载、解析、分块
将原始文档转换为可供嵌入的结构化文本块

主要模块：
- cos_service: 文件存储服务（COS + 本地回退）
- file_parser: 多种文件格式解析器
- text_cleaner: 文本清洗和预处理
- text_splitter: 基于LangChain的智能文本分割器
- fileddl_service: 文件清洗和分割服务入口
"""

from ingestion.cos_service import FileStorageService, get_cos_service
from ingestion.file_parser import (
    BaseParser,
    TextParser,
    MarkdownParser,
    PDFParser,
    CSVParser,
    WordParser,
    ExcelParser,
    HTMLParser,
    FileParserRegistry,
    get_file_parser,
)
from ingestion.text_cleaner import (
    TextCleaner,
    TextCleanOptions,
    TextPreprocessor,
    create_cleaner_from_options,
)
from ingestion.text_splitter import (
    TextSplitterFactory,
    SplitConfig,
    IntelligentTextSplitter,
    get_text_splitter,
    create_split_config_from_input,
)
from ingestion.fileddl_service import (
    FileDDLSplitService,
    get_file_ddl_split_service,
    file_ddl_split_service,
)

__all__ = [
    # 存储服务
    'FileStorageService',
    'get_cos_service',
    # 文件解析器
    'BaseParser',
    'TextParser',
    'MarkdownParser',
    'PDFParser',
    'CSVParser',
    'WordParser',
    'ExcelParser',
    'HTMLParser',
    'FileParserRegistry',
    'get_file_parser',
    # 文本清洗
    'TextCleaner',
    'TextCleanOptions',
    'TextPreprocessor',
    'create_cleaner_from_options',
    # 文本分割
    'TextSplitterFactory',
    'SplitConfig',
    'IntelligentTextSplitter',
    'get_text_splitter',
    'create_split_config_from_input',
    # 主服务
    'FileDDLSplitService',
    'get_file_ddl_split_service',
    'file_ddl_split_service',
]
