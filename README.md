# med-genAi_medical_chatbot
LLM based medical chatbot

## Features
- AI-powered medical information retrieval
- Integration with Google's Gemini model
- Vector storage using Pinecone
- Web interface using Flask

## ScreenShots
![image](https://github.com/user-attachments/assets/b774c93e-7a94-4787-904d-f840858d0405)
![image](https://github.com/user-attachments/assets/baa469dd-61dd-41f0-8f2c-ef2ef829d116)
![image](https://github.com/user-attachments/assets/ce99e87e-a89e-44a6-a168-cde376715ed0)
![image](https://github.com/user-attachments/assets/c5ad7ac1-7f75-4067-a375-9d4ed4cd091b)


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/med-genAi_medical_chatbot.git
   cd med-genAi_medical_chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   ```

4. Store your medical documents in the `Data/` directory

5. Run the index creation:
   ```bash
   python store_index.py
   ```

6. Start the application:
   ```bash
   python app.py
   ```

## License
Apache License 2.0
