import os
import time
import hashlib
import httpx
import logging

logger = logging.getLogger(__name__)

class HtmlParser:
    def __init__(self):
        self.base_url = os.getenv("BP_HTML_PARSER_URL")
        self.api_key = os.getenv("BP_HTML_PARSER_API_KEY")
        self.secret = os.getenv("BP_HTML_PARSER_SECRET")

    def parse(self, url: str) -> tuple[bool, str]:
        timestamp = str(int(time.time()))
        headers = {
            "X-API-KEY": self.api_key,
            "X-TIMESTAMP": timestamp,
            "X-SIGNATURE": (
                hashlib.sha256(
                    (self.api_key + timestamp + self.secret).encode()
                ).hexdigest()
            ),
        }
        try:
            with httpx.Client() as client:
                response = client.post(
                    self.base_url, json={"url": url}, headers=headers, timeout=30
                )
        except Exception as e:
            logger.warning(f"Failed to parser {url} with error: {e}")
            return False, ""
        if response.status_code == 200:
            response_data = response.json()
            page_content = response_data["content"]
            if 'extracted_text' in response_data:
                page_content = response_data['extracted_text']
            else:
                page_content = "No content found"
            return True, page_content
        else:
            logger.warning(
                f"Failed to parser {url} with status code {response.status_code}"
            )
            return False, ""
