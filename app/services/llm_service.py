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

    async def get_chat_response(self, user_query: str, retrieved_hadiths: list) -> str:
        # Create a detailed context from the database results
        # h[0] is Book, h[1] is Text. Ensure this line ends with ])
        context = "\n".join([f"Book: {h[0]}, Hadith: {h[1]}" for h in retrieved_hadiths])
        
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
        
        # Standard async generation
        response = await self.model.generate_content_async(prompt)
        return response.text