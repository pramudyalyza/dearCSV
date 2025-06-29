# 📄 Dear CSV

A user-friendly Streamlit application that allows you to upload CSV files and interact with your data using natural language queries powered by Google's Gemini AI.

## 🚀 Features

- **CSV Upload & Analysis**: Upload CSV files with automatic delimiter detection
- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent SQL Generation**: AI automatically generates optimized SQL queries
- **Interactive Results**: View query results in formatted tables
- **Smart Explanations**: Get natural language explanations of your data insights
- **Context-Aware Suggestions**: Receive relevant question suggestions based on your dataset

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Database**: SQLite (in-memory)
- **Data Processing**: Pandas

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dearCSV
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory just like `.env.example`:
   ```env
   GOOGLE_API_KEY=your_google_ai_api_key_here
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload your CSV file**
   - Use the file uploader to select your CSV file
   - The app automatically detects delimiters (comma, semicolon)
   - Preview the first 5 rows of your data

3. **Add context (optional)**
   - Provide context about your dataset to get more relevant suggestions
   - Example: "This is Q3 2025 sales data for Indonesia..."

4. **Start asking questions**
   - Use the suggested questions or ask your own

5. **Review results**
   - View the generated SQL query
   - See the results in a formatted table
   - Read the AI-generated explanation

![image](sequence_diagram.svg)

## 🔧 Customization

You can customize the application by modifying:
- **AI Model**: Change `MODEL_ID` in the code
- **System Prompts**: Adjust the LLM instructions
- **UI Styling**: Modify the Streamlit CSS

## 🚨 Limitations

- **File Size**: Large CSV files may impact performance
- **SQL Only**: Currently supports SELECT queries only
- **Memory Usage**: Data is stored in memory during the session