"""
Client script to test the FastMCP Quart application
"""

import asyncio
import json
import httpx
import os
import sys
import platform

def check_api_key():
    """Check if the ANTHROPIC_API_KEY environment variable is set and provide guidance if not."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("\nTo set your Anthropic API key:")
        
        if platform.system() == "Windows":
            print("\nFor Windows Command Prompt (CMD):")
            print("    set ANTHROPIC_API_KEY=your-api-key-here")
            print("\nFor Windows PowerShell:")
            print("    $env:ANTHROPIC_API_KEY=\"your-api-key-here\"")
        else:  # Linux, macOS, etc.
            print("\nFor Linux/macOS (bash, zsh, etc.):")
            print("    export ANTHROPIC_API_KEY=\"your-api-key-here\"")
            
        print("\nYou can get your API key from https://console.anthropic.com/")
        print("Add the command to your shell profile to make it persistent.")
        sys.exit(1)
    
    return api_key

async def main():
    # Check for API key first
    api_key = check_api_key()
    print(f"Using Anthropic API key: {api_key[:5]}...{api_key[-4:]}")
    
    # Define a sample conversation
    messages = [
        {"role": "user", "content": "What's 25 * 31? Can you also tell me the weather in New York?"}
    ]
    
    # Make a request to the chat endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:5111/chat",
            json={"messages": messages, "max_iterations": 3},
            timeout=60.0
        )
    
    # Parse and print the response
    if response.status_code == 200:
        result = response.json()
        print("Claude's response:")
        print(result["response"])
        
        print(f"\nTool calls used: {result.get('tool_calls_used', False)}")
        print(f"Iterations: {result.get('iterations', 0)}")
        
        print("\nFull conversation:")
        print(json.dumps(result["conversation"], indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    asyncio.run(main())
