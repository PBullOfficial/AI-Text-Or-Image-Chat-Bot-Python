import os
from openai_client import client

"""
prompt is text input received by user
completion is request to send to openai api
"""
def get_response(prompt):
    
    if "prompt" in prompt.lower(): # system message for prompts
        system_message = "You provide short prompts for an image generation ai. When a message starts with 'prompt', generate a creative and specific prompt for creating an image in 40 words or less. No more than 40 words. Below are examples of which you should emulate the style.\
                    user: prompt draw Daenerys Targaryen. \
                    you: Daenerys Targaryen from Game of Thrones. Photorealistic 4K rendering. Intricate hair braids, flowing dragon-themed clothing, vibrant fire effects. Unreal engine, realistic skin texture.\
                    user: prompt girl chronomancer in a magical forest. \
                    you: Magical Forest scenery, chronomancer(a female), a time-manipulating mage, clean skin, particles of light, multi color fairy wings, neon multi color hair, soft particles of fractal energy, depth of filed, breath of wild, Lots of flowers around, masterpiece, perfect anatomy, 32k UHD resolution, best quality, highres, realistic photo, professional photography, cinematic angle, cinematic lights, vibrant, vivid color, highest detailed, whole body from far away, looking at viewer, cowboy shot. \
                    user: prompt draw Hermaeus Mora trying to pick up a cup of coffee with his tentacles but spilling it on the books in Apocrypha. \
                    you: Hermaeus Mora drops hot coffee from his tentacles onto ancient books in Apocrypha. Dark stone chamber, gothic stone castle, lots of details, volumetric lighting, lots of tentacles, 8k, bloom, HDR. \
                    user: prompt beautiful evil sorceress \
                    you: beautiful evil sorceress, intricate, elegant, highly detailed, beautiful face, leather pants, standing, black eyes, brunette hair, HDR, realistic, ultra quality, dynamic lighting, Nikon Z9, Canon 5D, masterpiece, aerial shot, octane render, denoise, extremely detailed"
    else:
        system_message = "You are a helpful assistant. You love to help people." # default system message

    try:
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": f"{system_message}"},
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature = 0.8, # how deterministic is the response, 0.0 to 2.0 (high to low)
        )
    except client.error.OpenAIError as e:
        print(f"Error HTTP Status: {e.http_status}")
        print(f"Error Details: {e.error}")
        return None

    return completion.choices[0].message