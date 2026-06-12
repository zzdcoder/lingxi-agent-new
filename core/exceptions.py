"""
基础异常类

所有业务异常统一继承自 RAGException，便于上层统一捕获与序列化
"""


class RAGException(Exception):
    """项目根异常"""
    def __init__(self, message: str = "服务内部错误", code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class BusinessException(RAGException):
    """通用业务异常"""
    def __init__(self, message: str = "业务处理失败", code: int = 400):
        super().__init__(message, code)


class FileUploadException(RAGException):
    """文件上传相关异常"""
    def __init__(self, message: str = "文件上传失败", code: int = 400):
        super().__init__(message, code)


class FileProcessingException(RAGException):
    """文件处理相关异常"""
    def __init__(self, message: str = "文件处理失败", code: int = 400):
        super().__init__(message, code)


class FileDownloadException(RAGException):
    """文件下载相关异常"""
    def __init__(self, message: str = "文件下载失败", code: int = 400):
        super().__init__(message, code)


class FileParseException(RAGException):
    """文件解析相关异常"""
    def __init__(self, message: str = "文件解析失败", code: int = 400):
        super().__init__(message, code)


class FileSplitException(RAGException):
    """文件分割相关异常"""
    def __init__(self, message: str = "文件分割失败", code: int = 400):
        super().__init__(message, code)
