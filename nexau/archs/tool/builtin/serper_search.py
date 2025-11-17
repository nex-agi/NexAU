# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Any

import httpx


class SerperSearch:
    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Serper API key is required")
        self.base_url = "https://google.serper.dev/"
        self.timeout = timeout
        self.max_retries = max_retries
        self.result_key_for_type: dict = {
            "news": "news",
            "places": "places",
            "images": "images",
            "search": "organic",
        }

    def search(
        self,
        query: str,
        search_type: str = "search",
        num_results: int = 10,
    ) -> list[dict[str, Any]] | str:
        if search_type not in self.result_key_for_type.keys():
            return f"Invalid search type: {search_type}. Serper search type should be one of {self.result_key_for_type.keys()}"

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(
                    timeout=httpx.Timeout(
                        connect=self.timeout,
                        read=self.timeout,
                        write=self.timeout,
                        pool=self.timeout,
                    ),
                ) as client:
                    response = client.post(
                        self.base_url + search_type,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()

                    data = response.json()
                    results = data.get(
                        self.result_key_for_type[search_type],
                        [],
                    )
                    results = results[:num_results]
                    for result in results:
                        if "imageUrl" in result and result["imageUrl"].startswith(
                            "data:",
                        ):
                            # delete base64 image url
                            del result["imageUrl"]
                    return results

            except httpx.ConnectTimeout as e:
                if attempt == self.max_retries - 1:
                    return f"Connection timeout after {self.max_retries} attempts: {str(e)}"
                time.sleep(2**attempt)  # Exponential backoff
                continue

            except httpx.TimeoutException as e:
                if attempt == self.max_retries - 1:
                    return f"Request timeout after {self.max_retries} attempts: {str(e)}"
                time.sleep(2**attempt)
                continue

            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries - 1:
                    return f"HTTP error {e.response.status_code}: {str(e)}"
                time.sleep(2**attempt)
                continue

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Unexpected error: {str(e)}"
                time.sleep(2**attempt)
                continue

        return f"Failed to complete search after {self.max_retries} attempts"


if __name__ == "__main__":
    searcher = SerperSearch()

    results = searcher.search("腾讯游戏自主研发能力", "search", 5)
    print(results)
