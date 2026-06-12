"""
用户认证路由

提供验证码获取、用户注册、用户登录、当前用户信息查询等接口
"""

import base64

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.config import settings
from core.exceptions import BusinessException
from models.user_model import User
from models.user_schema import UserRegister, UserLogin, Token, UserOut, CaptchaResponse
from utils.auth import hash_password, verify_password, create_access_token, get_current_user
from utils.captcha import generate_captcha_text, create_captcha_image, store_captcha, verify_captcha

router = APIRouter(prefix="/auth", tags=["用户认证"])


@router.get("/captcha", summary="获取验证码")
async def get_captcha():
    """生成新的图片验证码，返回 Base64 Data URL"""
    text = generate_captcha_text()
    image_bytes = create_captcha_image(text)
    captcha_id = store_captcha(text)

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{b64_image}"

    return CaptchaResponse(captcha_id=captcha_id, image_url=image_url)


@router.post("/register", status_code=status.HTTP_201_CREATED, summary="用户注册")
async def register(request: UserRegister, db: AsyncSession = Depends(get_db)):
    """用户注册：校验验证码、检查用户名、bcrypt 哈希密码"""
    if not verify_captcha(request.captcha_id, request.captcha_text):
        raise BusinessException("验证码错误或已过期", code=400)

    result = await db.execute(select(User).where(User.username == request.username))
    existing = result.scalar_one_or_none()
    if existing is not None:
        raise BusinessException("用户名已存在", code=409)

    user = User(
        username=request.username,
        password_hash=hash_password(request.password),
        user_type=request.user_type,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserOut.model_validate(user)


@router.post("/login", summary="用户登录")
async def login(request: UserLogin, db: AsyncSession = Depends(get_db)):
    """用户登录：校验验证码、查询用户、验证密码、生成 JWT"""
    if not verify_captcha(request.captcha_id, request.captcha_text):
        raise BusinessException("验证码错误或已过期", code=400)

    result = await db.execute(select(User).where(User.username == request.username, User.deleted == 0))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.password_hash):
        raise BusinessException("用户名或密码错误", code=401)

    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "user_type": user.user_type,
    }
    access_token = create_access_token(data=token_data)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_expires_hours * 3600,
        user=UserOut.model_validate(user),
    )


@router.get("/me", summary="获取当前用户信息")
async def me(current_user: User = Depends(get_current_user)):
    """获取当前登录用户的基本信息，需携带 Authorization: Bearer <token>"""
    return UserOut.model_validate(current_user)
