import asyncio
import os
from services.screenshot_service import ScreenshotService
from dotenv import load_dotenv

load_dotenv()

async def test_real_url():
    # Use the URL provided by the user
    url = "https://www.monfric.ca/nouvelles/la-banque-du-canada-annonce-sa-decision-concernant-le-taux-directeur-du-10-decembre"
    
    print(f"Testing screenshot service with URL: {url}")
    
    try:
        service = ScreenshotService()
        screenshots = await service.get_embed_screenshots(url)
        
        print(f"Success! Found {len(screenshots)} embeds.")
        for i, s in enumerate(screenshots):
            print(f"Embed {i}: Type={s['type']}, Selector={s['selector']}, Screenshot size={len(s['screenshot'])} chars")
            
            # Optionally save one to check
            if i == 0:
                with open("test_screenshot_output.png", "wb") as f:
                    import base64
                    f.write(base64.b64decode(s['screenshot']))
                print("Saved first screenshot to test_screenshot_output.png")
                
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_url())
