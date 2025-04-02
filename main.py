import os
import json
import logging
import requests
import re
import hashlib
import asyncio
from markdown import markdown
from bs4 import BeautifulSoup

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get token from environment variable
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
SOY_TOKEN = os.environ.get("SOY_TOKEN")

# Dictionary to store conversation history for each user
user_sessions = {}
# Dictionary to store callback data mappings (to handle the 64-byte limit)
callback_data_map = {}
# Dictionary to track processing state
processing_requests = {}


LLM_START_PROMPT = """Ты - бот-консультант по подбору автомобиля сайта auto.ru.
Твоя задача - помочь пользователю выбрать автомобиль, который подходит его потребностям.
Задавай по одному вопросу за раз. Каждый вопрос должен содержать варианты ответов (не более 7 вариантов).
Обязательно включай вариант "Не знаю". Формат ответа: Текст вопроса [1. Вариант ответа] [2. Вариант ответа] и т.д.
Общайся с пользователем только на русском языке. Первым вопрос выясни, насколько хорошо пользователь разбирается в автомобилях
и сообщи ему, что он может писать текстом, если подходящих вариантов ответа нет.
После 5-6 вопросов предложи конкретную модель автомобиля, которая подходит по описанным критериям.
И сразу же спроси, что ему нравится в этой модели (или что не нравится), с предложенными вариантами ответов.
Если модель не подходит, надо предложить другие варианты. Когда пользователь будет доволен предложением, заверши диалог.
Помни, что ты представляешь интересы auto.ru"""

USER_START_PROMPT = "Привет! Я хочу подобрать автомобиль, но не знаю, что именно мне нужно. Помоги, пожалуйста."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    user_sessions[user_id] = []

    # Initial system prompt to set the context
    system_message = {
        "role": "system",
        "content": LLM_START_PROMPT
    }
    user_sessions[user_id].append(system_message)

    # Show the bot is typing
    await update.message.chat.send_action(ChatAction.TYPING)

    # First message to the user
    initial_prompt = {
        "role": "user",
        "content": USER_START_PROMPT
    }
    user_sessions[user_id].append(initial_prompt)

    # Get response from GPT
    response_text = await get_gpt_response(user_id)
    response_wo_think = remove_think_section(response_text)

    # Parse suggested answers and create buttons
    buttons = parse_options(response_wo_think)

    # Save assistant's response to session
    user_sessions[user_id].append({
        "role": "assistant",
        "content": response_text
    })

    intro = extract_question(response_wo_think)

    if buttons:
        keyboard = create_inline_keyboard(buttons, user_id)
        await update.message.reply_text(intro, reply_markup=keyboard)
    else:
        await update.message.reply_text(intro)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages."""
    user_id = update.effective_user.id
    user_message = update.message.text
    logger.info(f"Received message from user: {user_message}")

    # Check if there's already a request being processed for this user
    if user_id in processing_requests and processing_requests[user_id]:
        await update.message.reply_text("Пожалуйста, подождите. Я обрабатываю ваш предыдущий запрос...")
        return

    # Mark user as processing
    processing_requests[user_id] = True

    # Show typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        # Initialize session if not exists
        if user_id not in user_sessions:
            user_sessions[user_id] = []
            system_message = {
                "role": "system",
                "content": LLM_START_PROMPT
            }
            user_sessions[user_id].append(system_message)

        # Add user message to the conversation history
        user_sessions[user_id].append({
            "role": "user",
            "content": user_message
        })

        # Keep showing typing indicator while getting GPT response
        typing_task = asyncio.create_task(show_typing_repeatedly(update.message.chat))

        # Get response from GPT
        response_text = await get_gpt_response(user_id)
        response_wo_think = remove_think_section(response_text)

        # Parse suggested answers and create buttons
        buttons = parse_options(response_wo_think)

        # Save assistant's response to session
        user_sessions[user_id].append({
            "role": "assistant",
            "content": response_text
        })

        intro = extract_question(response_wo_think)

        if buttons:
            keyboard = create_inline_keyboard(buttons, user_id)
            await update.message.reply_text(intro, reply_markup=keyboard)
        else:
            await update.message.reply_text(intro)

    finally:
        # Mark user as not processing
        processing_requests[user_id] = False

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    user_id = update.effective_user.id
    callback_id = query.data

    # Check if there's already a request being processed for this user
    if user_id in processing_requests and processing_requests[user_id]:
        await query.answer("Пожалуйста, подождите. Я обрабатываю ваш предыдущий запрос...", show_alert=True)
        return

    # Mark user as processing
    processing_requests[user_id] = True

    # Immediately acknowledge the button press to show the user something happened
    await query.answer("Обрабатываю ваш выбор...")

    # Change the button text to show it's being processed
    original_keyboard = query.message.reply_markup

    # Add "⏳" to the pressed button to indicate processing
    new_keyboard = []

    for row in original_keyboard.inline_keyboard:
        new_row = []
        for button in row:
            if button.callback_data == callback_id:
                # selected_option = button.text
                new_row.append(InlineKeyboardButton(f"⏳ {button.text}", callback_data=button.callback_data))
            else:
                new_row.append(button)
        new_keyboard.append(new_row)

    # Update the message with the modified keyboard
    try:
        await query.edit_message_reply_markup(InlineKeyboardMarkup(new_keyboard))
    except Exception as e:
        logger.error(f"Error updating keyboard: {e}")

    try:
        # Retrieve the original option from the callback data map
        if callback_id in callback_data_map:
            user_message = callback_data_map[callback_id]
            logger.info(f"Resolved callback data: {callback_id} -> {user_message}")
        else:
            user_message = callback_id
            logger.warning(f"Callback data not found in map: {callback_id}")

        # Add user message to the conversation history
        user_sessions[user_id].append({
            "role": "user",
            "content": user_message
        })

        # Get response from GPT
        response_text = await get_gpt_response(user_id)
        response_wo_think = remove_think_section(response_text)

        # Parse suggested answers and create buttons
        buttons = parse_options(response_wo_think)

        # Save assistant's response to session
        user_sessions[user_id].append({
            "role": "assistant",
            "content": response_text
        })

        intro = extract_question(response_wo_think)

        try:
            if buttons:
                keyboard = create_inline_keyboard(buttons, user_id)
                await query.edit_message_text(text=intro, reply_markup=keyboard)
            else:
                await query.edit_message_text(text=intro)
        except Exception as e:
            logger.error(f"Error updating message: {e}")
            # Send a new message instead of editing
            if buttons:
                keyboard = create_inline_keyboard(buttons, user_id)
                await query.message.reply_text(text=intro, reply_markup=keyboard)
            else:
                await query.message.reply_text(text=intro)

    finally:
        # Mark user as not processing
        processing_requests[user_id] = False

async def get_gpt_response(user_id):
    """Get a response from LLM using the conversation history."""
    # url = "http://api.eliza.yandex.net/openai/v1/chat/completions"  # chatgpt
    url = "http://api.eliza.yandex.net/together/v1/chat/completions"  # deepseek

    payload = {
        # "model": "gpt-4o",
        "model": "deepseek-ai/deepseek-r1",
        "messages": user_sessions[user_id]
    }

    headers = {
        "authorization": f"OAuth {SOY_TOKEN}",
        "content-type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()

        if 'response' in response_json and 'choices' in response_json['response']:
            message_content = response_json['response']['choices'][0]['message']['content']
            logger.info(f"LLM response:\n{message_content}")
            return message_content
        else:
            logger.error(f"Unexpected response format: {response_json}")
            return "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."

    except Exception as e:
        logger.error(f"Error getting GPT response: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз."

async def show_typing_repeatedly(chat):
    """Show typing indicator repeatedly until the task is cancelled."""
    try:
        while True:
            await chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(3)  # Telegram typing indicator typically lasts ~3 seconds
    except asyncio.CancelledError:
        # Task was cancelled, which is expected
        pass

def remove_think_section(text):
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def markdown_to_plain_text(markdown_text):
    html = markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text()
    return plain_text

def extract_question(text):
    cleaned = markdown_to_plain_text(re.sub(r'\[.*?\]', '', text).strip())
    return cleaned

def parse_options(text):
    """Parse options from GPT response to create buttons."""
    options = []
    matches = re.findall(r'\[([^\]]+)\]', text)

    if matches:
        for match in matches:
            string = match.strip()
            if len(string) > 64:
                string = string[:64]
            options.append(string)
    return options

def create_inline_keyboard(options, user_id):
    """Create an inline keyboard with the given options, using hash IDs for callback data."""
    keyboard = []

    for option in options:
        # Create a unique ID for this option using hash
        hash_id = hashlib.md5(f"{user_id}:{option}".encode()).hexdigest()[:16]

        # Store the mapping
        callback_data_map[hash_id] = option

        # Create button with the hash ID as callback_data
        keyboard.append([InlineKeyboardButton(option, callback_data=hash_id)])

    return InlineKeyboardMarkup(keyboard)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the conversation."""
    user_id = update.effective_user.id
    user_sessions[user_id] = []

    # Clean up callback data map for this user
    to_remove = []
    for key in callback_data_map:
        if key.startswith(f"{user_id}:"):
            to_remove.append(key)

    for key in to_remove:
        del callback_data_map[key]

    await update.message.reply_text("Разговор сброшен. Давайте начнем заново. Используйте /start чтобы начать подбор автомобиля.")

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
