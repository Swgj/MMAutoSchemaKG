"""
Multimodal Tool Functions
Handles image downloading, caching, and encoding.
"""
import os
import hashlib
import base64
from curl_cffi import requests
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image

def get_cache_path(url: str, cache_dir: str = "images_cache") -> str:
    """Generate a unique file path for a URL using MD5 hash."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # 使用 URL 的哈希作为文件名，避免特殊字符问题
    hash_object = hashlib.md5(url.encode())
    filename = hash_object.hexdigest()
    # 默认存为 jpg，或者你可以先以此为名，下载后再确定后缀
    return os.path.join(cache_dir, f"{filename}.jpg")

def download_image(url: str, cache_dir: str = "images_cache") -> Optional[str]:
    """
    Download image from URL and save to cache directory.
    Returns the local file path if successful, None otherwise.
    """
    try:
        file_path = get_cache_path(url, cache_dir)
        
        # 1. 如果缓存已存在，直接返回路径
        if os.path.exists(file_path):
            return file_path
            
        # 2. 下载图片
        # 设置 User-Agent 避免被某些图床拦截
        headers = {
        "Referer": "https://www.google.com",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        response = requests.get(
            url, 
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        # 3. 验证是否真的是图片 (可选，但推荐)
        try:
            img = Image.open(BytesIO(response.content))
            img.verify() # 验证文件完整性
        except Exception:
            print(f"Warning: URL returned invalid image data: {url}")
            return None

        # 4. 写入文件
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        return file_path
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def image_to_base64(image_path: str) -> Optional[str]:
    """
    Read a local image file and convert it to Base64 string.
    """
    if not image_path or not os.path.exists(image_path):
        return None
        
    try:
        with open(image_path, "rb") as image_file:
            # 自动判断 MIME type 有点麻烦，这里偷懒默认为 jpeg
            # 如果需要更精确，可以用 mimetypes 库或 PIL
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def url_to_base64(url: str, cache_dir: str = "images_cache") -> Optional[str]:
    """
    Facade function: Convert URL directly to base64 (with caching).
    """
    # 1. 先尝试下载/获取缓存路径
    local_path = download_image(url, cache_dir)
    
    # 2. 如果获取成功，转 Base64
    if local_path:
        return image_to_base64(local_path)
    
    return None


def load_image_content_from_dict(image_id: str, image_dict: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Load image content from image_dict by image_id

    Args:
        image_id: The ID of the image
        image_dict: The dictionary of images

    Returns:
        The content of the image (OpenAI format)
    """
    img_url = image_dict.get(image_id)
    if img_url:
        base64_content = url_to_base64(img_url)
        if base64_content:
            text_content = {"type": "text", "text": f"Image {image_id} shows as follows:"}
            img_content = {"type": "image_url","image_url": {"url": base64_content}}
            return {
                "role": "user",
                "content": [text_content, img_content]
            }
    return None
