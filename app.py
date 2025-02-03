import asyncio
import json
import os
from typing import List

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

#lets use the ollam deepseek

URL_TO_SCRAPE = "https://x.com/sama"

INSTRUCTION_TO_LLM = "Extract all the tweets present in the page (only 10) and return them in a JSON format. Each tweet should have the following fields: 'author', 'content', 'date', put null if the field is not present. The tweets should be ordered by date in descending order."


class Product(BaseModel):
    name: str
    price: str


async def main():

    llm_strategy = LLMExtractionStrategy(
        provider="ollama/deepseek-r1",
        # api_token=os.getenv("DEEPSEEK_API"),
        # schema=Product.model_json_schema(),
        extraction_type="schema",
        instruction=INSTRUCTION_TO_LLM,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 800},
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True,
    )

    browser_cfg = BrowserConfig(headless=True, verbose=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:

        result = await crawler.arun(url=URL_TO_SCRAPE, config=crawl_config)
        print("The result is ", result)
        #lets store the response in txt file which is properly formatted
        with open('response.txt', 'w') as f:
            f.write(result.extracted_content)
        if result.success:
            data = json.loads(result.extracted_content)

            print("Extracted items:", data)

            llm_strategy.show_usage()
        else:
            print("Error:", result.error_message)


if __name__ == "__main__":
    asyncio.run(main())