import re
import google.generativeai as genai

class LLMService:
    def __init__(self):
        # 1. Import settings to get your API key
        from app.core.config import settings
        self.api_key = settings.GOOGLE_API_KEY
        
        # 2. Setup the Gemini API
        genai.configure(api_key=self.api_key)
        
        # 3. Using the standard stable model
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _is_greeting(self, user_query: str) -> bool:
        normalized = re.sub(r"[^a-z0-9 ]+", "", user_query.lower()).strip()
        tokens = normalized.split()
        if not tokens:
            return False

        exact_greetings = {
            "hi",
            "hello",
            "hey",
            "salam",
            "assalamualaikum",
            "assalamu alaikum",
            "good morning",
            "good afternoon",
            "good evening",
        }

        if normalized in exact_greetings:
            return True

        first_token = tokens[0]
        if len(tokens) <= 3 and first_token in {"hi", "hello", "hey", "salam", "assalamualaikum", "assalamu", "good"}:
            return True

        return False

    async def get_chat_response(self, user_query: str, retrieved_hadiths: list) -> str:
        if self._is_greeting(user_query):
            return (
                "Wa alaikum assalam! I am Himak, your Islamic Hadith assistant. "
                "Ask me about authentic Hadith and I will answer from the sources."
            )

        # Create a detailed context from the database results
        context = "\n".join([f"Book: {h.book}, Hadith: {h.text}" for h in retrieved_hadiths])
        
        prompt = f"""
        You are 'Himak', an expert Islamic AI Assistant.
        Your task is to answer the user's question based ONLY on the authentic Hadiths provided below.

        Guidelines:
        - If the answer is not in the context, say you don't know.
        - Cite the Book name for every Hadith you use.

        Context Hadiths:
        {context}

        User Question: {user_query}
        """
        
        try:
            # Standard async generation
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                return "I apologize, but my AI service has exceeded its daily free tier quota limits. Please try asking again tomorrow or provide a new API key."
            return f"I apologize, I encountered an error connecting to my AI brain. Detail: {str(e)}"