# SPDX-License-Identifier: Apache-2.0
"""
web_fetch tool - Fetches and processes content from URLs.

Based on gemini-cli's web-fetch.ts implementation.
Supports URL parsing, GitHub raw URL conversion, and fallback fetching.
"""

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlparse

# Configuration constants
URL_FETCH_TIMEOUT_MS = 10000
MAX_CONTENT_LENGTH = 100000


def _parse_prompt(text: str) -> tuple[list[str], list[str]]:
    """
    Parse a prompt to extract valid URLs and identify malformed ones.

    Returns:
        Tuple of (valid_urls, errors)
    """
    tokens = text.split()
    valid_urls = []
    errors = []

    for token in tokens:
        if not token:
            continue

        # Check if token appears to be a URL
        if "://" in token:
            try:
                parsed = urlparse(token)

                # Allowlist protocols
                if parsed.scheme in ("http", "https"):
                    valid_urls.append(token)
                else:
                    errors.append(f'Unsupported protocol in URL: "{token}". Only http and https are supported.')
            except Exception:
                errors.append(f'Malformed URL detected: "{token}".')

    return valid_urls, errors


def _is_private_ip(url: str) -> bool:
    """Check if URL points to a private/local IP address."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False

        # Check for localhost
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return True

        # Try to resolve and check if IP is private
        try:
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private or ip_obj.is_loopback
        except (socket.gaierror, ValueError):
            # Can't resolve - treat as potentially private
            return hostname.endswith(".local") or hostname.startswith("192.168.")

    except Exception:
        return False


def _convert_github_url(url: str) -> str:
    """Convert GitHub blob URL to raw URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def _fetch_url_content(url: str, timeout_ms: int = URL_FETCH_TIMEOUT_MS) -> dict[str, Any]:
    """
    Fetch content from a URL using requests.

    Returns dict with 'content', 'content_type', or 'error'.
    """
    try:
        import requests

        timeout_sec = timeout_ms / 1000.0

        response = requests.get(
            url,
            timeout=timeout_sec,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GeminiCLI/1.0)"},
        )

        response.raise_for_status()

        return {
            "content": response.text,
            "content_type": response.headers.get("content-type", ""),
            "status": response.status_code,
        }

    except ImportError:
        return {"error": "requests library not installed"}
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {timeout_ms}ms"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def _html_to_text(html_content: str) -> str:
    """Convert HTML to plain text."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        return text

    except ImportError:
        # Fallback: simple regex-based HTML stripping
        text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def web_fetch(
    prompt: str,
    fetch_function: Any | None = None,
) -> dict[str, Any]:
    """
    Processes content from URL(s) embedded in a prompt.

    Supports up to 20 URLs. Handles GitHub blob URL conversion automatically.
    Falls back to direct HTTP fetch if primary method fails or for private IPs.

    Args:
        prompt: Prompt containing URL(s) and instructions for processing
        fetch_function: Optional external fetch function for testing/mocking

    Returns:
        Dict with llmContent and returnDisplay matching gemini-cli format
    """
    try:
        # Validate prompt
        if not prompt or not prompt.strip():
            return {
                "llmContent": "The 'prompt' parameter cannot be empty and must contain URL(s) and instructions.",
                "returnDisplay": "Error: Empty prompt.",
                "error": {
                    "message": "The 'prompt' parameter cannot be empty.",
                    "type": "INVALID_PROMPT",
                },
            }

        # Parse URLs from prompt
        valid_urls, errors = _parse_prompt(prompt)

        if errors:
            error_msg = "Error(s) in prompt URLs:\n- " + "\n- ".join(errors)
            return {
                "llmContent": error_msg,
                "returnDisplay": "Error: Invalid URLs in prompt.",
                "error": {
                    "message": error_msg,
                    "type": "INVALID_URL",
                },
            }

        if not valid_urls:
            return {
                "llmContent": "The 'prompt' must contain at least one valid URL (starting with http:// or https://).",
                "returnDisplay": "Error: No valid URLs found.",
                "error": {
                    "message": "No valid URLs found in prompt.",
                    "type": "NO_URLS_FOUND",
                },
            }

        # Process the first URL (primary support)
        url = valid_urls[0]

        # Convert GitHub URL if needed
        url = _convert_github_url(url)

        # Check if it's a private IP
        is_private = _is_private_ip(url)

        # Use external fetch function if provided
        if fetch_function and not is_private:
            try:
                result = fetch_function(prompt)

                if isinstance(result, dict):
                    response_text = result.get("text", "")
                    sources = result.get("sources", [])
                    grounding_supports = result.get("groundingSupports", [])

                    # Check for processing errors
                    if not response_text.strip() and not sources:
                        # Fall through to fallback
                        pass
                    else:
                        # Process sources if available
                        modified_response = response_text
                        source_list = []

                        if sources:
                            for idx, source in enumerate(sources):
                                title = source.get("web", {}).get("title", "Untitled")
                                uri = source.get("web", {}).get("uri", "Unknown URI")
                                source_list.append(f"[{idx + 1}] {title} ({uri})")

                            if grounding_supports:
                                insertions = []
                                for support in grounding_supports:
                                    segment = support.get("segment", {})
                                    chunk_indices = support.get("groundingChunkIndices", [])

                                    if segment and chunk_indices:
                                        marker = "".join(f"[{i + 1}]" for i in chunk_indices)
                                        insertions.append(
                                            {
                                                "index": segment.get("endIndex", 0),
                                                "marker": marker,
                                            }
                                        )

                                insertions.sort(key=lambda x: x["index"], reverse=True)
                                chars = list(modified_response)
                                for ins in insertions:
                                    idx = min(ins["index"], len(chars))
                                    chars.insert(idx, ins["marker"])
                                modified_response = "".join(chars)

                            if source_list:
                                modified_response += "\n\nSources:\n" + "\n".join(source_list)

                        return {
                            "llmContent": modified_response,
                            "returnDisplay": "Content processed from prompt.",
                        }

            except Exception:
                # Fall through to fallback
                pass

        # Fallback: Direct HTTP fetch
        fetch_result = _fetch_url_content(url)

        if "error" in fetch_result:
            error_msg = f"Error during fallback fetch for {url}: {fetch_result['error']}"
            return {
                "llmContent": f"Error: {error_msg}",
                "returnDisplay": f"Error: {error_msg}",
                "error": {
                    "message": error_msg,
                    "type": "WEB_FETCH_FALLBACK_FAILED",
                },
            }

        # Process content
        content = fetch_result["content"]
        content_type = fetch_result.get("content_type", "")

        # Convert HTML to text if needed
        if "text/html" in content_type.lower() or not content_type:
            text_content = _html_to_text(content)
        else:
            text_content = content

        # Truncate if too long
        text_content = text_content[:MAX_CONTENT_LENGTH]

        # Build response
        llm_content = f"""Content fetched from {url}:

---
{text_content}
---

This content was fetched directly from the URL. Please use it to respond to the original request: "{prompt}"
"""

        return {
            "llmContent": llm_content,
            "returnDisplay": f"Content for {url} processed using fallback fetch.",
        }

    except Exception as e:
        prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
        error_msg = f'Error processing web content for prompt "{prompt_preview}": {str(e)}'
        return {
            "llmContent": f"Error: {error_msg}",
            "returnDisplay": f"Error: {error_msg}",
            "error": {
                "message": error_msg,
                "type": "WEB_FETCH_PROCESSING_ERROR",
            },
        }
