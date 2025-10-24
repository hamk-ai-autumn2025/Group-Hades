import google.generativeai as genai
import os
import mimetypes
from pathlib import Path
from PIL import Image
import io

def setup_api_key():
    """
    Configures the generative AI library with the API key from 
    environment variables.
    """
    try:
        api_key = os.environ["GOOGLE_API_KEY"]
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        print("API key configured successfully.")
    except KeyError:
        print("Error: GOOGLE_API_KEY environment variable not found.")
        print("Please set the environment variable before running the script.")
        print("Example (Linux/macOS): export GOOGLE_API_KEY='YOUR_API_KEY'")
        print("Example (Windows): set GOOGLE_API_KEY='YOUR_API_KEY'")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()

def get_image_parts(image_paths):
    """
    Loads images from file paths and prepares them for the API.
    """
    image_parts = []
    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Image file not found at '{path_str}'. Skipping.")
            continue
        
        try:
            img = Image.open(path)
            

            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = f"image/{img.format.lower()}"
                
            print(f"Loading image: {path_str} (MIME type: {mime_type})")

            img_bytes = path.read_bytes()
            
            image_parts.append({
                "mime_type": mime_type,
                "data": img_bytes
            })
        except Exception as e:
            print(f"Error loading image '{path_str}': {e}. Skipping.")
            
    return image_parts

def get_user_inputs():
    """
    Collects the text prompt and image file paths from the user.
    """
    print("\n--- Product Description Generator ---")

    user_text = input("Enter your product instructions (e.g., 'What is this product? Write a catchy slogan.'): \n")
    if not user_text:
        user_text = "Analyze these images and write a compelling product description and 3 marketing slogans."

    image_paths = []
    while True:
        path = input("Enter the file path for an image (or press Enter to finish): ")
        if not path:
            if not image_paths:
                print("You must provide at least one image.")
                continue
            else:
                break
        image_paths.append(path.strip())
        
    return user_text, image_paths

def generate_description(user_text, image_parts):
    """
    Generates content using the Gemini model.
    """
    if not image_parts:
        print("No valid images were loaded. Cannot generate description.")
        return

    system_prompt = """
You are an expert e-commerce copywriter. Your responses must be energetic, persuasive, and professional. 
You will be given images of a product and a user's instructions.
Your job is to synthesize this information into:
1.  A compelling product title.
2.  A 2-3 sentence product description.
3.  A bulleted list of 3-5 key features.
4.  A short, catchy marketing slogan.

Format your response clearly using markdown. Do not add any extra conversation or preamble.
"""

    model = genai.GenerativeModel(
        'gemini-2.5-flash', # 
        system_instruction=system_prompt
    )
    # --- End of Change ---

    prompt_parts = [user_text] + image_parts

    print("\nGenerating description... This may take a moment.")
    
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f"\nAn error occurred while calling the API: {e}")
        return None

def main():
    setup_api_key()
    
    user_text, image_paths = get_user_inputs()
    
    image_parts = get_image_parts(image_paths)
    
    description = generate_description(user_text, image_parts)
    
    if description:
        print("\n--- Generated Product Content ---")
        print(description)
        print("---------------------------------")

if __name__ == "__main__":
    main()
