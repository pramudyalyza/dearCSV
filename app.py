import os
import json
import sqlite3
import logging
import traceback
import pandas as pd
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import BlockedPromptException, StopCandidateException, GenerationConfig

# --- load environment variables ---
load_dotenv()

# --- Global Constants ---
GENERIC_TABLE_NAME = "uploadedData"

# --- Streamlit Configuration ---
st.set_page_config(page_title="Dear CSV", page_icon="📄", layout="wide")

# --- Helper Functions ---
def setup():
    st.markdown(
        """
        <style>
               /* Remove blank space at top and bottom */ 
               .block-container {
                   padding-top: 1rem;
                   padding-bottom: 0rem;
                }

        </style>
        """,
        unsafe_allow_html=True,)

    st.title("📄 Dear CSV")

    st.markdown(
        """
        This app allows you to upload a CSV file and interact with it using a Large Language Model (LLM). 
        You can ask questions about your data, and the LLM will generate SQL queries to retrieve the information you need.
        """)

    st.markdown(
    """
    <style>
        .block-container {
            padding-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True)

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def create_schema_string(df, table_name):
    schema = []
    for column, dtype in df.dtypes.items():
        column_type = str(dtype).upper()
        if column_type == "OBJECT":
            column_type = "STRING"
        elif column_type.startswith("INT"):
            column_type = "INTEGER"
        elif column_type.startswith("FLOAT"):
            column_type = "REAL"
        elif column_type.startswith("DATETIME"):
            column_type = "TIMESTAMP"
        schema.append(
            {
                "name": column,
                "type": column_type
            }
        )

    table_data = {
        "table": [
            {"table_name": table_name, "schema": schema}
        ]
    }
    return json.dumps(table_data, indent=2)

def generate_prompt_suggestions(schema: str, context: str) -> list[str]:
    """
    Generates 3 sample questions based on schema and context.
    """
    system_prompt = f"""
    You are an expert data analyst. Your task is to help a user explore their dataset by suggesting interesting and relevant questions.

    You will be provided with the table schema and user-provided context about the dataset.

    Based on all this information, generate 3 sample questions that the user could ask.

    Guidelines:
    - The questions must be answerable using a standard SQL query on the given schema.
    - The questions should be insightful and encourage exploration of the data.
    - Tailor the questions to the specific columns and data types in the schema.
    - If the user has provided context, use it to make the questions more relevant.
    - Return ONLY a JSON array of strings, where each string is a suggested question.
    - Do not include any other text, explanation, or markdown formatting.

    Example Output:
    [
        "What is the total revenue for each product category?",
        "Which month had the highest number of new users?",
        "List the top 3 cities with the most sales."
    ]
    """

    user_prompt = f"""
    Here is the information to base your suggestions on:

    **Table Schema:**
    ```json
    {schema}
    ```

    **User-Provided Dataset Context:**
    "{context if context else 'No context provided.'}"

    Please generate 3 relevant questions based on the data above.
    """

    try:
        model = genai.GenerativeModel(
            model_name=st.session_state["MODEL_ID"],
            system_instruction=system_prompt,
        )
        response = model.generate_content(user_prompt)

        clean_response = (
            response.text.strip().replace("```json", "").replace("```", "")
        )

        suggestions = json.loads(clean_response)
        if isinstance(suggestions, list):
            return suggestions[:3] if len(suggestions) > 3 else suggestions
        else:
            return []
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error generating suggestions: {e}")
        if "response" in locals() and response.text:
            # Fallback: if JSON fails, try to parse lines as suggestions
            return [
                line.strip()
                for line in response.text.split("\n")
                if line.strip() and line.strip().endswith("?")
            ][:3] # Limit fallback too
        return []

def execute_query_tool(query: str) -> str:
    try:
        conn = st.session_state["db_conn"]
        result_df = pd.read_sql_query(query, conn)
        # Return as JSON-serializable structure
        return json.dumps({"result": result_df.to_dict()})
    except Exception as e:
        error_message = f"Sqlite Query Error: {str(e)}"
        st.error(f"Error in execute_query_tool: {error_message}")
        return json.dumps({"Sqlite Query error": error_message})
    
def format_sql(query: str) -> str:
    """Adds newlines to SQL queries for better readability"""
    # Insert newline after key clauses
    formatted = query.replace("SELECT", "SELECT\n    ") \
                    .replace("FROM", "\nFROM") \
                    .replace("WHERE", "\nWHERE") \
                    .replace("GROUP BY", "\nGROUP BY") \
                    .replace("ORDER BY", "\nORDER BY") \
                    .replace("JOIN", "\nJOIN") \
                    .replace("AND", "\n    AND") \
                    .replace("OR", "\n    OR")
    return formatted

def generate_explanation(question: str, query: str, result_df: pd.DataFrame) -> str:
    """Generate natural language explanation of query results"""
    system_prompt = """
    You are a data analyst assistant. Your task is to explain SQL query results to users in simple terms.
    You will be given:
    1. The user's original question
    2. The SQL query used to answer the question
    3. The query results (as a DataFrame)
    
    Your explanation should:
    - Be concise and easy to understand
    - Highlight key insights from the data
    - Explain trends or patterns in the results
    - Avoid technical jargon
    - Never make up facts not present in the results
    
    Example:
    User question: "How many males and females are in the data?"
    Query: "SELECT gender, COUNT(*) FROM patients GROUP BY gender"
    Results: 
        gender  count
        Male    45
        Female  55
    
    Explanation: "The data shows we have more female patients (55) than male patients (45), with females making up about 55% of the total."
    """
    
    # Convert results to string for LLM input
    result_str = result_df.head().to_markdown(index=False)
    
    prompt = f"""
    **User Question**: {question}
    **Executed Query**: 
    ```sql
    {query}
    ```
    **Query Results**:
    {result_str}
    """
    
    model = genai.GenerativeModel(
        model_name=st.session_state["MODEL_ID"],
        system_instruction=system_prompt
    )
    response = model.generate_content(prompt)
    return response.text

def humanize_list(items):
    if not items:
        return ""
    return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

# Main Application Logic
def main():
    # Initialize conversation_history at the start
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Initialize prompt_suggestions if not already in session state
    if "prompt_suggestions" not in st.session_state:
        st.session_state.prompt_suggestions = []

    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV here:", type=["csv"])

    if uploaded_file is not None:
        # Check if this is a new file
        if "uploaded_file_name" not in st.session_state or st.session_state["uploaded_file_name"] != uploaded_file.name:
            st.session_state["uploaded_file_name"] = uploaded_file.name
            # Clear relevant session states
            keys_to_reset = [
                "df",
                "schema",
                "table_name",
                "db_conn",
                "llm_model_chat_session",
                "user_dataset_context", 
                "context_status", 
                "prompt_suggestions",
                "suggestion_prompt",
                "conversation_started"
            ]

            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            # Reset conversation history for new file
            st.session_state.conversation_history = []  # Reset instead of delete
            logging.info("New CSV file uploaded. Resetting session.")

        # Read CSV into DataFrame only if not already in session state
        if "df" not in st.session_state:
            try:
                # Read 1000 rows to infer separator, then reread if necessary
                temp_df = pd.read_csv(uploaded_file, nrows=1000) 
                if temp_df.shape[1] == 1 and uploaded_file.name.endswith('.csv'):
                    uploaded_file.seek(0) # Reset file pointer for full read
                    df = pd.read_csv(uploaded_file, sep=';')
                    if df.shape[1] > 1:
                        st.info("Detected semicolon as separator.")
                    else: # If semicolon also results in one column, revert to comma
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file)
                else: # If first attempt (comma) worked or not a CSV, use it
                    uploaded_file.seek(0) # Reset file pointer for full read
                    df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='latin1') # Try a different encoding
            except pd.errors.ParserError as e:
                st.error(f"Failed to parse CSV: {e}. Please check file format.")
                uploaded_file.seek(0) # Reset pointer
                # Fallback to comma if parsing fails, or manual inspection.
                # A more robust solution might involve letting user choose separator.
                df = pd.read_csv(uploaded_file) 
            except Exception as e:
                st.error(f"An unexpected error occurred while reading CSV: {e}")
                st.stop() # Critical error, stop execution.

            st.session_state["df"] = df


        st.session_state["table_name"] = GENERIC_TABLE_NAME
        st.session_state["schema"] = create_schema_string(
            st.session_state["df"], st.session_state["table_name"]
        )

        st.write("First 5 rows:")
        st.dataframe(st.session_state["df"].head())

        if "context_status" not in st.session_state:
            st.session_state.context_status = None  # None | "submitted"
        
        if st.session_state.context_status != "submitted":
            user_context = st.text_area(
                "Add context about your dataset",
                placeholder="e.g. This is Q3 2025 sales data for Indonesia...",
                height=68,
                key="context_input",
            )

            # Button to submit context and generate initial suggestions
            if st.button("Submit"):
                st.session_state["user_dataset_context"] = user_context.strip()
                st.session_state.context_status = "submitted" # Mark context as submitted

                # Generate suggestions only once after context submission
                with st.spinner("Analyzing data and generating initial questions..."):
                    st.session_state.prompt_suggestions = generate_prompt_suggestions(
                        schema=st.session_state["schema"],
                        context=st.session_state.get("user_dataset_context", ""),
                    )
                st.toast("Context saved and suggestions generated", icon="✅")

        if st.session_state.context_status == "submitted":
            context = st.session_state.get("user_dataset_context", "")

            if "db_conn" not in st.session_state:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                st.session_state["df"].to_sql(
                    st.session_state["table_name"],
                    conn,
                    if_exists="replace",
                    index=False,
                )
                st.session_state["db_conn"] = conn
                logging.info("In-memory SQLite database provider initialized.")

            llm_system_instruction = f"""
            You are an expert in answering questions users have about their data stored in sqlite3.
            Your job is to generate relevant SQL statements and call the `execute_query_tool` to get the best answer.

            You have access to the following table in sqlite3: {st.session_state["table_name"]}.
            Here is its schema: {st.session_state["schema"]}.

            Additional context from user: {context}

            Instructions:
            1. When a user asks a question, formulate the most appropriate SQL query
            2. Call the `execute_query_tool` with the query
            3. Present ONLY the final results to the user (DO NOT generate explanations)

            Important guidelines:
            - Use meaningful aliases for column names
            - Order results for clarity when appropriate
            - Select only necessary columns; avoid SELECT *
            - Use valid sqlite3 SQL (only SELECT statements)
            - Handle datetimes with: strftime('%Y-%m-%d', date_column)
            - Only use fields listed in the provided schema
            - If query fails, analyze the error and attempt a corrected version
            """

            # Initialize LLM model chat session if not already done or if a new file was uploaded
            if "llm_model_chat_session" not in st.session_state:
                model = genai.GenerativeModel(
                    model_name=st.session_state["MODEL_ID"],
                    tools=[execute_query_tool],
                    system_instruction=llm_system_instruction,
                    generation_config=GenerationConfig(temperature=0.3),
                )
                st.session_state["llm_model_chat_session"] = model.start_chat(
                    history=[]
                )
                logging.info("LLM chat session initialized.")

            # --- Display Conversation History ---
            for chat_entry in st.session_state.conversation_history:
                role = chat_entry["role"]
                avatar = "🤖" if role == "model" else None

                with st.chat_message(role, avatar=avatar):
                    if role == "user":
                        st.write(chat_entry["content"])
                    else:  # model
                        if "queries" in chat_entry:
                            # Display queries
                            st.markdown("**Generated Query:**")
                            for query in chat_entry["queries"]:
                                st.markdown(f"```sql\n{query}\n```")

                            # Display results
                            if "result_dfs" in chat_entry and chat_entry["result_dfs"]:
                                st.markdown("**Query Result:**")
                                for df_dict in chat_entry["result_dfs"]:
                                    st.dataframe(pd.DataFrame(df_dict), hide_index=True)

                        if "explanation" in chat_entry and chat_entry["explanation"].strip():
                            st.markdown("**Explanation:**")
                            st.markdown(chat_entry["explanation"])
                        elif "content" in chat_entry and chat_entry["content"].strip():
                            st.markdown(chat_entry["content"])


            # --- Display Prompt Suggestions (only if available) ---
            # These will only be generated once after initial context submission
            if st.session_state.prompt_suggestions and not st.session_state.conversation_started:
                st.markdown("---")
                st.markdown("### Suggested Questions")
                clicked_suggestion = None
                
                cols = st.columns(3)
                for i, suggestion in enumerate(st.session_state.prompt_suggestions[:3]):
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                            clicked_suggestion = suggestion
                            st.session_state.conversation_started = True  # Mark conversation started
                
                if clicked_suggestion:
                    st.session_state.suggestion_prompt = clicked_suggestion
                    
                st.markdown("---") # Visual separator after suggestions
            else:
                st.markdown("---")

            # --- Chat Input and Processing ---
            prompt_from_chat = st.chat_input("Ask a question about your data...")

            if "suggestion_prompt" in st.session_state and st.session_state.suggestion_prompt:
                st.session_state.conversation_started = True
                prompt_to_use = st.session_state.suggestion_prompt
                del st.session_state.suggestion_prompt
            else:
                prompt_to_use = prompt_from_chat

            if prompt_to_use: # Proceed if there's a prompt
                st.session_state.conversation_history.append({"role": "user", "content": prompt_to_use})
                with st.chat_message("user"):
                    st.write(prompt_to_use)

                with st.chat_message("model", avatar="🤖"):
                    try:
                        response = st.session_state["llm_model_chat_session"].send_message(prompt_to_use)

                        # Initialize variables
                        queries = []
                        result_dfs = []
                        function_called = False

                        # Process all response parts
                        for part in response.parts:
                            if part.function_call and part.function_call.name == "execute_query_tool":
                                function_called = True
                                if 'query' in part.function_call.args:
                                    query_str = part.function_call.args['query']
                                    queries.append(query_str)

                                    # Execute the query and get result
                                    tool_response = execute_query_tool(query_str)
                                    result_json = json.loads(tool_response)

                                    if "result" in result_json:
                                        # Convert result string to DataFrame
                                        result_content = result_json["result"]
                                        # Ensure result_content is a dict for pd.DataFrame
                                        if isinstance(result_content, dict):
                                            result_df = pd.DataFrame(result_content)
                                            result_dfs.append(result_df)
                                        else:
                                            st.error("Unexpected result format returned from query tool.")
                                            # Log unexpected result for debugging
                                            logging.error(f"Unexpected tool result type: {type(result_content)}")
                                            continue
                                    elif "Sqlite Query error" in result_json:
                                        st.error(f"Query Error: {result_json['Sqlite Query error']}")
                                        # Capture error for history
                                        history_entry = {
                                            "role": "model",
                                            "content": f"Error: {result_json['Sqlite Query error']}",
                                            "queries": queries # Still show the attempted query
                                        }
                                        st.session_state.conversation_history.append(history_entry)
                                        st.stop() # Stop further processing for this prompt

                        # Display components
                        if queries:
                            st.markdown("**Generated Query:**")
                            for q in queries:
                                formatted_q = format_sql(q)
                                st.markdown(f"```sql\n{formatted_q}\n```")

                        if result_dfs:
                            st.markdown("**Query Result:**")
                            for df_display in result_dfs:
                                st.dataframe(df_display, hide_index=True)

                        # Store in conversation history
                        history_entry = {
                            "role": "model",
                            "queries": queries,
                            # Convert DataFrame to dictionary list for serialization
                            "result_dfs": [df.to_dict(orient="list") for df in result_dfs],
                            "content": "" # Initialize content for explanation
                        }

                        if queries and result_dfs:
                            try:
                                # Use the last successful query/result for explanation
                                explanation = generate_explanation(
                                    question=prompt_to_use,
                                    query=queries[-1],
                                    result_df=result_dfs[-1]
                                )

                                st.markdown(explanation)
                                history_entry["explanation"] = explanation # Store explanation
                            except Exception as e:
                                st.error(f"Failed to generate explanation: {str(e)}")
                                history_entry["explanation"] = "Explanation unavailable" # Indicate failure

                        # Handle cases where LLM might not call tool (e.g., direct answer)
                        if not function_called and response.text:
                            st.markdown(response.text)
                            history_entry["content"] = response.text

                        st.session_state.conversation_history.append(history_entry)

                    except BlockedPromptException as e:
                        st.warning(f"Response blocked: {e.block_reason}. Please try rephrasing your question.")
                        st.session_state.conversation_history.append({"role": "model", "content": "Response blocked by safety filters. Please try rephrasing."})
                    except StopCandidateException as e:
                        st.error(f"LLM generation stopped prematurely. Finish reason: {e.candidate.finish_reason}. Debug info: {e.candidate.content}")
                        st.warning("The LLM stopped generating a full response. This can happen due to safety settings or unexpected output.")
                        st.session_state.conversation_history.append({"role": "model", "content": "The AI stopped generating a full response. Please try rephrasing."})
                    except Exception as e:
                        error_msg = f"An unexpected error occurred during LLM interaction: {str(e)}"
                        st.error(error_msg)
                        st.session_state.conversation_history.append({
                            "role": "model",
                            "content": error_msg
                        })
                        st.text(traceback.format_exc())

    else:
        st.info("Please upload a CSV file to begin chatting with your data.")
        # When no file is uploaded, reset all relevant session states
        keys_to_reset_on_no_file = [
            "uploaded_file_name",
            "df",
            "schema",
            "table_name",
            "db_conn",
            "user_dataset_context",
            "conversation_history",
            "llm_model_chat_session",
            "context_status",
            "prompt_suggestions",
            "suggestion_prompt",
            "conversation_started"]
        
        for key in keys_to_reset_on_no_file:
            if key in st.session_state:
                del st.session_state[key]


if __name__ == "__main__":
    setup()
    
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        st.error("Please set your GOOGLE_API_KEY environment variable")
        st.stop()
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    if "MODEL_ID" not in st.session_state:
        st.session_state["MODEL_ID"] = "gemini-2.5-flash-preview-05-20" 
    
    main()