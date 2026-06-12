"""
图片验证码工具

使用 Pillow 生成干扰型验证码图片，在服务端内存中存储验证码文本
生产环境建议替换为 Redis
"""

import io
import random
import string
import time
import base64
from threading import RLock

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from core.config import settings

_captcha_store: dict[str, dict] = {}
_captcha_lock = RLock()


def generate_captcha_id() -> str:
    """生成随机的验证码唯一标识符"""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=16))


def generate_captcha_text(length: int = 4) -> str:
    """生成随机验证码文本，去除易混淆字符"""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choices(chars, k=length))


def _get_font(size: int):
    """获取字体对象，优先使用系统字体"""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def create_captcha_image(text: str, width: int = 140, height: int = 50) -> bytes:
    """生成干扰型验证码图片并返回 PNG 字节数据"""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 背景噪点
    for _ in range(300):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
        draw.point((x, y), fill=color)

    # 干扰线
    for _ in range(5):
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height)
        x2 = random.randint(width // 2, width)
        y2 = random.randint(0, height)
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

    # 逐个绘制字符
    font = _get_font(28)
    char_width = width // len(text)
    for i, char in enumerate(text):
        color = (random.randint(30, 120), random.randint(30, 120), random.randint(30, 120))
        angle = random.randint(-30, 30)
        char_img = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((5, 2), char, font=font, fill=color)
        rotated = char_img.rotate(angle, expand=1, resample=Image.BICUBIC)
        paste_x = i * char_width + random.randint(2, 8)
        paste_y = random.randint(2, 10)
        image.paste(rotated, (paste_x, paste_y), rotated)

    # 轻微模糊
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def store_captcha(text: str) -> str:
    """将验证码文本存入内存，返回唯一标识符"""
    captcha_id = generate_captcha_id()
    expires_at = time.time() + settings.captcha_expires_seconds
    with _captcha_lock:
        _captcha_store[captcha_id] = {"text": text, "expires_at": expires_at}
    return captcha_id


def verify_captcha(captcha_id: str, captcha_text: str) -> bool:
    """校验用户提交的验证码是否正确，校验后立即删除"""
    with _captcha_lock:
        record = _captcha_store.get(captcha_id)
        if record is None:
            return False
        if time.time() > record["expires_at"]:
            del _captcha_store[captcha_id]
            return False
        valid = record["text"].upper() == captcha_text.upper()
        del _captcha_store[captcha_id]
        return valid


def cleanup_expired_captchas() -> int:
    """清理所有已过期的验证码记录"""
    now = time.time()
    removed = 0
    with _captcha_lock:
        expired_ids = [cid for cid, rec in _captcha_store.items() if now > rec["expires_at"]]
        for cid in expired_ids:
            del _captcha_store[cid]
            removed += 1
    return removed
