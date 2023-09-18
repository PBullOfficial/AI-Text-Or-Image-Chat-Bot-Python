import discord
from discord.ext import commands
from text_by_api import get_response
from image_by_api import get_image
import os
import logging
import requests
import io

# debug logging
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# access discord bot token environment variable
DISCORD_BOT_TOKEN = os.getenv("YOUR_DISCORD_BOT_TOKEN")

# define bot intents
intents = discord.Intents.default() 
intents.typing = False # can adjust these based on bot's needs
intents.presences = False
intents.message_content = True # enable message content intent

# initialize bot
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.event
async def on_message(message):
    if message.author == bot.user: # ignore messages from bot itself
        return

    # text controller
    if message.content.startswith('!ask'):
        # get prompt from message content
        prompt = message.content[5:].strip()
        # get response from API
        response = get_response(prompt)
        # extract only content message from API response
        api_content = response["content"]
        if len(api_content) >= 2000: # paginate response if over Discord's character limit
            await send_paginated_message(message.channel, api_content)
        else:
            await message.reply(api_content)

    # image controller
    if message.content.startswith('!draw'):
        prompt = message.content[5:].strip()
        response = get_image(prompt)
        
        if response.startswith("https"): # check for valid URL
            img_url = response

            # download image from the URL
            img_response = requests.get(img_url)

            if img_response.status_code == 200: # if request successful
                img_bytes = img_response.content
                img_file = io.BytesIO(img_bytes) # create file-like binary stream object for discord.File to send (expects file-like object)

                # send image URL as Discord file attachment
                await message.reply(file=discord.File(img_file, "output.png"))

            else: # if request unsuccessful
                await message.reply("Failed to fetch the image.")

        else:
            await message.reply("Image URL not found in the response.")

# split response message if over Discord's 2000 character limit
async def send_paginated_message(channel, text):
    max_chars = 2000
    start = 0 # index 0 of text

    # iterate through text in chunks of 2000 characters
    while start < len(text):
        end = start + max_chars # end index of each chunk
        if end > len(text):
            end = len(text)  # prevent out of bounds error

        # escape / and > characters before sending
        chunk = text[start:end]
        chunk = chunk.replace('/', '\/')  # replace / with \/
        chunk = chunk.replace('>', '\>')  # replace > with \>

        if text[end:end + 1] == '\0':  # check for null character
            await channel.send(chunk)
            return
        else:
            await channel.send(chunk)
            start = end # update start index for next chunk

# run bot
bot.run(f"{DISCORD_BOT_TOKEN}", log_handler=handler)

