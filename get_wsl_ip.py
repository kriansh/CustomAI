#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper script to find WSL2 IP address and update the .env file
"""

import os
import subprocess
import re
from dotenv import load_dotenv, set_key

def get_wsl_ip():
    """Get IP address of WSL2 instance"""
    try:
        # Run the command to get WSL IP
        result = subprocess.run(
            ["wsl", "hostname", "-I"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        # Extract the first IP address (usually the correct one)
        ip_match = re.search(r"\d+\.\d+\.\d+\.\d+", result.stdout)
        if ip_match:
            return ip_match.group(0)
        else:
            print("Couldn't extract IP from WSL output")
            return None
    except Exception as e:
        print(f"Error getting WSL IP: {e}")
        return None

def update_env_file(ip_address):
    """Update .env file with WSL IP address"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    # Load current .env file
    load_dotenv(env_path)
    
    # Update Ollama base URL
    ollama_url = f"http://{ip_address}:11434"
    print(f"Updating Ollama base URL to: {ollama_url}")
    
    # Update the .env file
    set_key(env_path, "OLLAMA_BASE_URL", ollama_url)
    print(f"Updated .env file at {env_path}")

if __name__ == "__main__":
    print("Finding WSL IP address...")
    wsl_ip = get_wsl_ip()
    
    if wsl_ip:
        print(f"WSL IP address found: {wsl_ip}")
        update_env_file(wsl_ip)
        print("\nNow you can run your chatbot with:")
        print("  python main.py chat --question=\"What programs does Aroma College offer?\"")
        print("  python main.py web")
    else:
        print("Could not find WSL IP. Make sure WSL is running.")
        print("You might need to manually update OLLAMA_BASE_URL in your .env file.")
