import asyncio
import os
import sys
from playwright.async_api import async_playwright

async def generate_auth():
    """
    Open a local headful browser, allow the user to log in, and save the session.
    """
    print("\n--- Playwright Session Generator ---")
    print("1. A browser window will open.")
    print("2. Log into Facebook, Instagram, and any other required sites.")
    print("3. Once logged in, come back here and press ENTER to save the session.\n")
    
    async with async_playwright() as p:
        # Launch LOCAL browser (not browserless) for manual login
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Navigate to a useful starting point
        await page.goto("https://www.facebook.com")
        
        print("Waiting for you to log in...")
        
        # In a real environment, we'd wait for user input. 
        # Since I can't wait indefinitely, I'll prompt the user.
        input("Press ENTER here after you have successfully logged in...")
        
        # Save the storage state
        auth_file = "auth.json"
        await context.storage_state(path=auth_file)
        print(f"\n✅ Success! Session saved to {auth_file}")
        
        await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(generate_auth())
    except EOFError:
        pass
    except KeyboardInterrupt:
        print("\nAborted.")
