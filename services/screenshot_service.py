import asyncio
import base64
import os
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

class ScreenshotService:
    def __init__(self, browserless_url: Optional[str] = None, browserless_token: Optional[str] = None):
        self.browserless_url = browserless_url or os.getenv("BROWSERLESS_URL")
        self.browserless_token = browserless_token or os.getenv("BROWSERLESS_TOKEN")
        self.auth_path = "auth.json"
        
        if not self.browserless_url:
            raise ValueError("BROWSERLESS_URL is not set")
            
        # Ensure the URL is in the correct format for CDP/WebSocket
        # If it's https://, convert to wss://
        ws_url = self.browserless_url
        if ws_url.startswith("http://"):
            ws_url = ws_url.replace("http://", "ws://", 1)
        elif ws_url.startswith("https://"):
            ws_url = ws_url.replace("https://", "wss://", 1)
        elif not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
            ws_url = f"wss://{ws_url}"
            
        # Ensure /chromium path is present for CDP
        if "/chromium" not in ws_url and "/playwright" not in ws_url:
            if ws_url.endswith("/"):
                ws_url += "chromium"
            else:
                ws_url += "/chromium"
                
        self.ws_endpoint = f"{ws_url}?token={self.browserless_token}" if self.browserless_token else ws_url
        
        # Add stealth mode for bypassing bot detection
        if "?" in self.ws_endpoint:
            self.ws_endpoint += "&stealth"
        else:
            self.ws_endpoint += "?stealth"
            
        print(f"Connecting to: {self.ws_endpoint}")

    def _transform_url(self, url: str) -> str:
        """
        Transform URLs to public embed versions to bypass login walls.
        """
        # Instagram
        if "instagram.com/p/" in url or "instagram.com/reels/" in url:
            if "/embed" not in url:
                # Remove trailing slash if exists and add /embed/
                base_url = url.rstrip('/')
                return f"{base_url}/embed/"
        
        # Facebook
        if "facebook.com" in url:
            if "/posts/" in url or "/photos/" in url or "/videos/" in url:
                import urllib.parse
                encoded_url = urllib.parse.quote(url)
                return f"https://www.facebook.com/plugins/post.php?href={encoded_url}&show_text=true"

        return url

    async def capture_single_embed(self, embed_url: str) -> Optional[Dict]:
        """
        Navigate directly to an embed URL and capture it.
        """
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(self.ws_endpoint)
            
            context_args = {
                "viewport": {'width': 800, 'height': 800},
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            if os.path.exists(self.auth_path):
                print(f"Using session from {self.auth_path}")
                context_args["storage_state"] = self.auth_path
                
            context = await browser.new_context(**context_args)
            page = await context.new_page()
            
            try:
                transformed_url = self._transform_url(embed_url)
                print(f"Navigating directly to embed: {transformed_url} (original: {embed_url})...")
                await page.goto(transformed_url, wait_until="load", timeout=90000)
                await page.wait_for_timeout(5000) # Wait for hydration
                
                # Detect the type of embed based on URL or content
                type_detected = "unknown"
                if "twitter.com" in embed_url or "x.com" in embed_url:
                    type_detected = "twitter"
                elif "youtube.com" in embed_url or "youtu.be" in embed_url:
                    type_detected = "youtube"
                elif "instagram.com" in embed_url:
                    type_detected = "instagram"
                elif "tiktok.com" in embed_url:
                    type_detected = "tiktok"

                # Try to find a good element to screenshot, or default to full page
                # For Twitter/X, we might want the tweet article
                selector = "body"
                if type_detected == "twitter":
                    selector = "article"
                elif type_detected == "youtube":
                    selector = "#player"
                
                element = await page.query_selector(selector) if selector != "body" else None
                
                if element:
                    await element.scroll_into_view_if_needed()
                    await page.wait_for_timeout(1000)
                    screenshot_bytes = await element.screenshot(timeout=10000)
                else:
                    # Fallback to full page screenshot of the viewport
                    screenshot_bytes = await page.screenshot(full_page=False, timeout=10000)
                
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                
                return {
                    "type": type_detected,
                    "url": embed_url,
                    "screenshot": screenshot_base64
                }
                
            except Exception as e:
                print(f"Error capturing single embed {embed_url}: {e}")
                return None
            finally:
                await browser.close()

    async def get_embed_screenshots(self, url: str) -> List[Dict]:
        screenshots = []
        
        async with async_playwright() as p:
            # Connect to Browserless
            browser = await p.chromium.connect_over_cdp(self.ws_endpoint)
            
            context_args = {
                "viewport": {'width': 1280, 'height': 800},
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            if os.path.exists(self.auth_path):
                print(f"Using session from {self.auth_path}")
                context_args["storage_state"] = self.auth_path
                
            context = await browser.new_context(**context_args)
            page = await context.new_page()
            
            try:
                # Navigate to the article
                transformed_url = self._transform_url(url)
                print(f"Navigating to {transformed_url}...")
                await page.goto(transformed_url, wait_until="load", timeout=90000)
                
                # Wait for content to settle
                await page.wait_for_timeout(10000)
                
                # Define selectors for common embeds
                selectors = {
                    "twitter": "blockquote.twitter-tweet, iframe[id^='twitter-widget-'], twitter-widget",
                    "youtube": "iframe[src*='youtube.com/embed'], iframe[src*='youtu.be'], lite-youtube",
                    "instagram": "blockquote.instagram-media, iframe.instagram-media, .instagram-media",
                    "tiktok": "blockquote.tiktok-embed, iframe[src*='tiktok.com/embed'], .tiktok-embed",
                    "general_iframe": "iframe.embed-content, .embed-container iframe, .video-container iframe, .entry-content iframe",
                    "iframe_all": "iframe"
                }
                
                # Filter iframes by size if using iframe_all
                processed_selectors = list(selectors.items())
                
                for embed_type, selector in processed_selectors:
                    if browser.is_connected() is False:
                        print("Browser disconnected, stopping.")
                        break
                        
                    elements = await page.query_selector_all(selector)
                    print(f"Found {len(elements)} targets for {embed_type}")
                    
                    for i, element in enumerate(elements[:15]): 
                        try:
                            # Scroll into view to trigger lazy loading / hydration
                            await element.scroll_into_view_if_needed()
                            await page.wait_for_timeout(3000) # Give it time to hydrate/load
                            
                            # Check visibility and box
                            box = await element.bounding_box()
                            if not box or box['width'] < 50 or box['height'] < 50:
                                print(f"Element {embed_type} index {i} too small or no box: {box}")
                                continue
                                
                            is_visible = await element.is_visible()
                            if not is_visible:
                                print(f"Element {embed_type} index {i} not visible after scroll")
                                continue

                            # Take screenshot
                            print(f"Capturing screenshot for {embed_type} index {i}...")
                            screenshot_bytes = await element.screenshot(timeout=10000)
                            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                            
                            screenshots.append({
                                "type": embed_type,
                                "selector": selector,
                                "index": i,
                                "screenshot": screenshot_base64
                            })
                            print(f"Successfully captured {embed_type} index {i}")
                            
                        except Exception as e:
                            print(f"Failed to capture screenshot for {embed_type} index {i}: {e}")
                            if "closed" in str(e).lower():
                                return screenshots
                            
            except Exception as e:
                print(f"Error during screenshot process: {e}")
            finally:
                await browser.close()
                
        return screenshots

if __name__ == "__main__":
    # Quick test
    async def test():
        service = ScreenshotService()
        test_url = "https://example.com" # Replace with a real one for manual testing
        results = await service.get_embed_screenshots(test_url)
        print(f"Captured {len(results)} screenshots")
        
    # asyncio.run(test())
