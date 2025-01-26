import nest_asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


nest_asyncio.apply()


API_TOKEN = "7757837093:AAGgXJlyZXkzWqU0GAn-FI_EjKd10_TYa1M"


print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on device: {device}")

async def start(update: Update, context):
  
    await update.message.reply_text("Hello! I am your AI assistant. How can I help you today?")

async def process_message(update: Update, context):
   
    try:
        user_input = update.message.text
        print(f"User input received: {user_input}")
        

        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        print("Input tokenized.")
        

        outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
        print("Model inference completed.")
        

        ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated response: {ai_response}")
        

        await update.message.reply_text(ai_response)
    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text("Oops! Something went wrong. Please try again.")


app = ApplicationBuilder().token(API_TOKEN).build()


app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
