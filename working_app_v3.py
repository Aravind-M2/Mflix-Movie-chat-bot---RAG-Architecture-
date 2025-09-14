import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine
import re
import json

# LangChain & RAG imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# --- Config ---
st.set_page_config(page_title="Movie RAG Chat", page_icon="🎬", layout="wide")
st.markdown("""
    <style>
    .main { padding: 0rem 2rem; }
    .stChatMessage { border-radius: 12px; padding: 8px 12px; margin-bottom: 6px; }
    .user-msg { background-color: #DCF8C6; text-align: right; }
    .assistant-msg { background-color: #F1F0F0; text-align: left; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

# --- Database Setup ---
openai_api_key = "api key"
db_user = "postgres"
db_password = "12345678"
db_host = "localhost"
db_port = "5432"
db_name = "mflix"

DB_URI = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(DB_URI)

# --- Load FAISS & LLM ---
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
faiss_index = FAISS.load_local("faiss_movie_index", embedding_model, allow_dangerous_deserialization=True)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

# --- Helper Functions ---
def query_postgres(sql):
    with engine.begin() as conn:
        return pd.read_sql_query(sql, conn)

def classify_query_type(query):
    prompt = f"""
    You are an intelligent query classifier for a movie database. Your task is to determine if a given query can be answered by directly querying the database's structure (tables and columns) or if it requires a deeper, meaning-based understanding (like recommendations or similarity).

    Classify the query into one of two types:
    1. Structured
    2. Semantic

    **Criteria for Structured Queries:**
    A query is **Structured** if it asks for specific information that can be directly found or calculated from the provided database tables and their columns. This includes, but is not limited to:
    - Retrieving values from specific columns (e.g., director, cast, plot, title, year, language, country).
    - Filtering data based on column values (e.g., movies released in a certain year, movies with a specific rating, movies by a particular director).
    - Aggregations (e.g., counts of movies, average ratings, total runtime).
    - Questions involving specific entities and their attributes (e.g., "director of X movie", "actors in Y movie", "genres of Z movie").

    **Criteria for Semantic Queries:**
    A query is **Semantic** if it requires understanding context, meaning, or relationships beyond direct table lookups. This typically involves:
    - Recommendations (e.g., "recommend a movie similar to...", "movies I might like").
    - Open-ended descriptive questions that are not direct column lookups (e.g., "tell me about the plot of X movie" *if* the intent is a summary beyond just returning the 'plot' column, though for 'plot' column lookup, it's structured).
    - Queries involving subjective interpretation or reasoning (e.g., "what's a good movie?", "movies that make you think").
    - Questions about similarity or relatedness that aren't direct join operations.

    Here is the data sample from the database for your understanding of the available structure:


    --- Table: theaters ---
    _id | theaterId | street1 | street2 | state | city | zipcode | type | Latitude | Longitude
    ------------------------------------------------------------------------------------------
    59a47286cfa9a3a73e51e72f | 1004 | 5072 Pinnacle Sq | NaN | AL | Birmingham | 35235 | Point | -86.642662 | 33.605438
    59a47286cfa9a3a73e51e738 | 1015 | 1721 Osgood Dr | NaN | PA | Altoona | 16602 | Point | -78.382912 | 40.490524
    59a47286cfa9a3a73e51e73a | 1019 | 390 Northridge Mall | NaN | CA | Salinas | 93906 | Point | -121.65946 | 36.715809

    --- Table: users ---
    _id | name | email | password
    -----------------------------
    59b99db4cfa9a34dcd7885b6 | Ned Stark | sean_bean@gameofthron.es | $2b$12$UREFwsRUoyF0CRqGNK0LzO0HM/jLhgUCNNIJ9RJAqMUQ74crlJ1Vu
    59b99db5cfa9a34dcd7885b8 | Jaime Lannister | nikolaj_coster-waldau@gameofthron.es | $2b$12$6vz7wiwO.EI5Rilvq1zUc./9480gb1uPtXcahDxIadgyC3PS8XCUK
    59b99db9cfa9a34dcd7885bf | Jon Snow | kit_harington@gameofthron.es | $2b$12$fDEu1Ru66tLWAVidMN.b0.929BlfnyqdGuhWMyzfOAf/ATYOyLoY6

    --- Table: comments ---
    _id | name | email | movie_id | text | date
    -------------------------------------------
    5a9427648b0beebeb6957aa3 | Yolanda Owen | yolanda_owen@fakegmail.com | 573a1391f29313caabcd6d40 | Occaecati commodi quidem aliquid delectus dolores. Facilis fugiat soluta maxime ipsum. Facere quibusdam vitae eius in fugit voluptatum beatae. | 1980-07-13 06:41:13
    5a9427648b0beebeb6957abd | John Bishop | john_bishop@fakegmail.com | 573a1391f29313caabcd6f98 | Accusamus qui distinctio ut ab saepe tenetur. Quae optio aut eius deleniti veritatis error. Eligendi ducimus rerum recusandae doloribus. Natus quisquam expedita voluptatum voluptatibus natus quidem. | 1972-04-16 14:52:53
    5a9427648b0beebeb6957bdd | Theresa Holmes | theresa_holmes@fakegmail.com | 573a1391f29313caabcd8979 | Unde ut eum doloremque expedita commodi exercitationem. Error soluta temporibus quasi. Libero quam nulla mollitia officia ipsa. Odio harum cupiditate a dignissimos. | 2003-06-25 15:21:24

    --- Table: movies ---
    _id | plot | runtime | poster | title | fullplot | released | rated | lastupdated | year | type | num_mflix_comments | metacritic | year_raw | plot_embedding
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    573a1390f29313caabcd446f | A greedy tycoon decides, on a whim, to corner the world market in wheat. This doubles the price of bread, forcing the grain's producers into charity lines and further into poverty. The film... | 14.0 | NaN | A Corner in Wheat | A greedy tycoon decides, on a whim, to corner the world market in wheat. This doubles the price of bread, forcing the grain's producers into charity lines and further into poverty. The film continues to contrast the ironic differences between the lives of those who work to grow the wheat and the life of the man who dabbles in its sale for profit. | 1909-12-13 | G | 2015-08-13 00:46:30.660000 | 1909 | movie | 1 | None | 1909 | None
    573a1390f29313caabcd4803 | Cartoon figures announce, via comic strip balloons, that they will move - and move they do, in a wildly exaggerated style. | 7.0 | https://m.media-amazon.com/images/M/MV5BYzg2NjNhNTctMjUxMi00ZWU4LWI3ZjYtNTI0NTQxNThjZTk2XkEyXkFqcGdeQXVyNzg5OTk2OA@@._V1_SY1000_SX677_AL_.jpg | Winsor McCay, the Famous Cartoonist of the N.Y. Herald and His Moving Comics | Cartoonist Winsor McCay agrees to create a large set of drawings that will be photographed and made into a motion picture. The job requires plenty of drawing supplies, and the cartoonist must also overcome some mishaps caused by an assistant. Finally, the work is done, and everyone can see the resulting animated picture. | 1911-04-08 | NaN | 2015-08-29 01:09:03.030000 | 1911 | movie | 0 | None | 1911 | None
    573a1390f29313caabcd4eaf | A woman, with the aid of her police officer sweetheart, endeavors to uncover the prostitution ring that has kidnapped her sister, and the philanthropist who secretly runs it. | 88.0 | https://m.media-amazon.com/images/M/MV5BYzk0YWQzMGYtYTM5MC00NjM2LWE5YzYtMjgyNDVhZDg1N2YzXkEyXkFqcGdeQXVyMzE0MjY5ODA@._V1_SY1000_SX677_AL_.jpg | Traffic in Souls | NaN | 1913-11-24 | TV-PG | 2015-09-15 02:07:14.247000 | 1913 | movie | 1 | None | 1913 | None

    --- Table: writers ---
    id | movie_id | writer
    ----------------------
    1 | 573a1390f29313caabcd42e8 | Unknown
    2 | 573a1390f29313caabcd446f | Unknown
    3 | 573a1390f29313caabcd4803 | Winsor McCay (comic strip "Little Nemo in Slumberland")

    --- Table: directors ---
    id | movie_id | director
    ------------------------
    1 | 573a1390f29313caabcd42e8 | Edwin S. Porter
    2 | 573a1390f29313caabcd446f | D.W. Griffith
    3 | 573a1390f29313caabcd4803 | Winsor McCay

    --- Table: cast ---
    id | movie_id | cast_member
    ---------------------------
    1 | 573a1390f29313caabcd42e8 | A.C. Abadie
    2 | 573a1390f29313caabcd42e8 | Gilbert M. 'Broncho Billy' Anderson
    3 | 573a1390f29313caabcd42e8 | George Barnes

    --- Table: genres ---
    id | movie_id | genre
    ---------------------
    1 | 573a1390f29313caabcd42e8 | Short
    2 | 573a1390f29313caabcd42e8 | Western
    3 | 573a1390f29313caabcd446f | Short

    --- Table: languages ---
    id | movie_id | language
    ------------------------
    1 | 573a1390f29313caabcd42e8 | English
    2 | 573a1390f29313caabcd446f | English
    3 | 573a1390f29313caabcd4803 | English

    --- Table: countries ---
    id | movie_id | country
    -----------------------
    1 | 573a1390f29313caabcd42e8 | USA
    2 | 573a1390f29313caabcd446f | USA
    3 | 573a1390f29313caabcd4803 | USA

    --- Table: tomatoes ---
    id | movie_id | viewer_rating | viewer_numreviews | viewer_meter | critic_rating | critic_numreviews | critic_meter | boxoffice | consensus | fresh | rotten | lastupdated | production | website | dvd_release
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1 | 573a1390f29313caabcd42e8 | 3.7 | 2559 | 75 | 7.6 | 6 | 100 | Unknown | No consensus available | 6 | 0 | 2015-08-08 19:16:10 | Unknown | Unknown | None
    2 | 573a1390f29313caabcd446f | 3.6 | 109 | 73 | nan | 0 | 0 | Unknown | No consensus available | 0 | 0 | 2015-05-11 18:36:53 | Unknown | Unknown | None
    3 | 573a1390f29313caabcd4803 | 3.4 | 89 | 47 | nan | 0 | 0 | Unknown | No consensus available | 0 | 0 | 2015-08-20 18:51:24 | Unknown | Unknown | None

    --- Table: imdb ---
    movie_id | imdb_rating | imdb_votes | imdb_id
    ---------------------------------------------
    573a1390f29313caabcd42e8 | 7.4 | 9847 | 439
    573a1390f29313caabcd446f | 6.6 | 1375 | 832
    573a1390f29313caabcd4803 | 7.3 | 1034 | 1737

    --- Table: awards ---
    movie_id | award_wins | award_nominations | award_text
    ------------------------------------------------------
    573a1390f29313caabcd42e8 | 1 | 0 | 1 win.
    573a1390f29313caabcd446f | 1 | 0 | 1 win.
    573a1390f29313caabcd4803 | 1 | 0 | 1 win.

    just a note: mention cast inside dounle cots if you have to use the table name or cast as postgresql will mistake it to a key word if not used within double cots.
    Query: "{query}"
    Answer:
    """
    response = llm.predict(prompt)
    return "structured" if "structured" in response.lower() else "semantic"


def handle_structured_query(query):
    prompt = f"""
    Given this user query, write a PostgreSQL query for the tables in mflix database :
    understand the user query and use appropriate column names to write the query, example if user mentions movie name in question, you should be able to understand that title column in the movies table is what user could be reffering.
    here is the sample data for you to understand the structure and content of the tables in mflix database:
    
    --- Table: theaters ---
    _id | theaterId | street1 | street2 | state | city | zipcode | type | Latitude | Longitude
    ------------------------------------------------------------------------------------------
    59a47286cfa9a3a73e51e72f | 1004 | 5072 Pinnacle Sq | NaN | AL | Birmingham | 35235 | Point | -86.642662 | 33.605438
    59a47286cfa9a3a73e51e738 | 1015 | 1721 Osgood Dr | NaN | PA | Altoona | 16602 | Point | -78.382912 | 40.490524
    59a47286cfa9a3a73e51e73a | 1019 | 390 Northridge Mall | NaN | CA | Salinas | 93906 | Point | -121.65946 | 36.715809

    --- Table: users ---
    _id | name | email | password
    -----------------------------
    59b99db4cfa9a34dcd7885b6 | Ned Stark | sean_bean@gameofthron.es | $2b$12$UREFwsRUoyF0CRqGNK0LzO0HM/jLhgUCNNIJ9RJAqMUQ74crlJ1Vu
    59b99db5cfa9a34dcd7885b8 | Jaime Lannister | nikolaj_coster-waldau@gameofthron.es | $2b$12$6vz7wiwO.EI5Rilvq1zUc./9480gb1uPtXcahDxIadgyC3PS8XCUK
    59b99db9cfa9a34dcd7885bf | Jon Snow | kit_harington@gameofthron.es | $2b$12$fDEu1Ru66tLWAVidMN.b0.929BlfnyqdGuhWMyzfOAf/ATYOyLoY6

    --- Table: comments ---
    _id | name | email | movie_id | text | date
    -------------------------------------------
    5a9427648b0beebeb6957aa3 | Yolanda Owen | yolanda_owen@fakegmail.com | 573a1391f29313caabcd6d40 | Occaecati commodi quidem aliquid delectus dolores. Facilis fugiat soluta maxime ipsum. Facere quibusdam vitae eius in fugit voluptatum beatae. | 1980-07-13 06:41:13
    5a9427648b0beebeb6957abd | John Bishop | john_bishop@fakegmail.com | 573a1391f29313caabcd6f98 | Accusamus qui distinctio ut ab saepe tenetur. Quae optio aut eius deleniti veritatis error. Eligendi ducimus rerum recusandae doloribus. Natus quisquam expedita voluptatum voluptatibus natus quidem. | 1972-04-16 14:52:53
    5a9427648b0beebeb6957bdd | Theresa Holmes | theresa_holmes@fakegmail.com | 573a1391f29313caabcd8979 | Unde ut eum doloremque expedita commodi exercitationem. Error soluta temporibus quasi. Libero quam nulla mollitia officia ipsa. Odio harum cupiditate a dignissimos. | 2003-06-25 15:21:24

    --- Table: movies ---
    _id | plot | runtime | poster | title | fullplot | released | rated | lastupdated | year | type | num_mflix_comments | metacritic | year_raw | plot_embedding
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    573a1390f29313caabcd446f | A greedy tycoon decides, on a whim, to corner the world market in wheat. This doubles the price of bread, forcing the grain's producers into charity lines and further into poverty. The film... | 14.0 | NaN | A Corner in Wheat | A greedy tycoon decides, on a whim, to corner the world market in wheat. This doubles the price of bread, forcing the grain's producers into charity lines and further into poverty. The film continues to contrast the ironic differences between the lives of those who work to grow the wheat and the life of the man who dabbles in its sale for profit. | 1909-12-13 | G | 2015-08-13 00:46:30.660000 | 1909 | movie | 1 | None | 1909 | None
    573a1390f29313caabcd4803 | Cartoon figures announce, via comic strip balloons, that they will move - and move they do, in a wildly exaggerated style. | 7.0 | https://m.media-amazon.com/images/M/MV5BYzg2NjNhNTctMjUxMi00ZWU4LWI3ZjYtNTI0NTQxNThjZTk2XkEyXkFqcGdeQXVyNzg5OTk2OA@@._V1_SY1000_SX677_AL_.jpg | Winsor McCay, the Famous Cartoonist of the N.Y. Herald and His Moving Comics | Cartoonist Winsor McCay agrees to create a large set of drawings that will be photographed and made into a motion picture. The job requires plenty of drawing supplies, and the cartoonist must also overcome some mishaps caused by an assistant. Finally, the work is done, and everyone can see the resulting animated picture. | 1911-04-08 | NaN | 2015-08-29 01:09:03.030000 | 1911 | movie | 0 | None | 1911 | None
    573a1390f29313caabcd4eaf | A woman, with the aid of her police officer sweetheart, endeavors to uncover the prostitution ring that has kidnapped her sister, and the philanthropist who secretly runs it. | 88.0 | https://m.media-amazon.com/images/M/MV5BYzk0YWQzMGYtYTM5MC00NjM2LWE5YzYtMjgyNDVhZDg1N2YzXkEyXkFqcGdeQXVyMzE0MjY5ODA@._V1_SY1000_SX677_AL_.jpg | Traffic in Souls | NaN | 1913-11-24 | TV-PG | 2015-09-15 02:07:14.247000 | 1913 | movie | 1 | None | 1913 | None

    --- Table: writers ---
    id | movie_id | writer
    ----------------------
    1 | 573a1390f29313caabcd42e8 | Unknown
    2 | 573a1390f29313caabcd446f | Unknown
    3 | 573a1390f29313caabcd4803 | Winsor McCay (comic strip "Little Nemo in Slumberland")

    --- Table: directors ---
    id | movie_id | director
    ------------------------
    1 | 573a1390f29313caabcd42e8 | Edwin S. Porter
    2 | 573a1390f29313caabcd446f | D.W. Griffith
    3 | 573a1390f29313caabcd4803 | Winsor McCay

    --- Table: cast ---
    id | movie_id | cast_member
    ---------------------------
    1 | 573a1390f29313caabcd42e8 | A.C. Abadie
    2 | 573a1390f29313caabcd42e8 | Gilbert M. 'Broncho Billy' Anderson
    3 | 573a1390f29313caabcd42e8 | George Barnes

    --- Table: genres ---
    id | movie_id | genre
    ---------------------
    1 | 573a1390f29313caabcd42e8 | Short
    2 | 573a1390f29313caabcd42e8 | Western
    3 | 573a1390f29313caabcd446f | Short

    --- Table: languages ---
    id | movie_id | language
    ------------------------
    1 | 573a1390f29313caabcd42e8 | English
    2 | 573a1390f29313caabcd446f | English
    3 | 573a1390f29313caabcd4803 | English

    --- Table: countries ---
    id | movie_id | country
    -----------------------
    1 | 573a1390f29313caabcd42e8 | USA
    2 | 573a1390f29313caabcd446f | USA
    3 | 573a1390f29313caabcd4803 | USA

    --- Table: tomatoes ---
    id | movie_id | viewer_rating | viewer_numreviews | viewer_meter | critic_rating | critic_numreviews | critic_meter | boxoffice | consensus | fresh | rotten | lastupdated | production | website | dvd_release
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1 | 573a1390f29313caabcd42e8 | 3.7 | 2559 | 75 | 7.6 | 6 | 100 | Unknown | No consensus available | 6 | 0 | 2015-08-08 19:16:10 | Unknown | Unknown | None
    2 | 573a1390f29313caabcd446f | 3.6 | 109 | 73 | nan | 0 | 0 | Unknown | No consensus available | 0 | 0 | 2015-05-11 18:36:53 | Unknown | Unknown | None
    3 | 573a1390f29313caabcd4803 | 3.4 | 89 | 47 | nan | 0 | 0 | Unknown | No consensus available | 0 | 0 | 2015-08-20 18:51:24 | Unknown | Unknown | None

    --- Table: imdb ---
    movie_id | imdb_rating | imdb_votes | imdb_id
    ---------------------------------------------
    573a1390f29313caabcd42e8 | 7.4 | 9847 | 439
    573a1390f29313caabcd446f | 6.6 | 1375 | 832
    573a1390f29313caabcd4803 | 7.3 | 1034 | 1737

    --- Table: awards ---
    movie_id | award_wins | award_nominations | award_text
    ------------------------------------------------------
    573a1390f29313caabcd42e8 | 1 | 0 | 1 win.
    573a1390f29313caabcd446f | 1 | 0 | 1 win.
    573a1390f29313caabcd4803 | 1 | 0 | 1 win.
    
    Query: "{query}"

    Return only the SQL code, do not explain anything.
    """
    sql = llm.predict(prompt)
    try:
        df = query_postgres(sql)
        return df
        # return df.head(10).to_markdown()

    except Exception as e:
        return f"SQL Error: {e}\nGenerated SQL: {sql}"

def handle_semantic_query(query, faiss_index = faiss_index, llm = llm): # Added faiss_index and llm as parameters
    docs = faiss_index.similarity_search(query, k=5)

    # --- CRITICAL CHANGE HERE ---
    # Construct the context by explicitly including the title from metadata
    context_parts = []
    for doc in docs:
        movie_title = doc.metadata.get('title', 'Unknown Title')
        movie_plot = doc.page_content # This is the plot, as you stored it
        context_parts.append(f"Title: {movie_title}\nPlot: {movie_plot}")

    context = "\n---\n".join(context_parts)
    # --- END CRITICAL CHANGE ---

    # Extract the movie title from the query if present, for better contextualization
    movie_in_query_match = re.search(r'movie like "([^"]+)"|movie like (\w[\w\s]*\w)', query, re.IGNORECASE)
    movie_in_query = None
    if movie_in_query_match:
        movie_in_query = movie_in_query_match.group(1) or movie_in_query_match.group(2)
        if movie_in_query:
            movie_in_query = movie_in_query.strip().replace('"', '')

    prompt = f"""
    You are a helpful movie recommendation assistant. Based on the provided movie details (including titles and plot summaries), suggest movies that are similar to the movie mentioned in the user's query.

    Here are some relevant movie details:
    {context}

    ---

    **Instructions for your answer:**
    1. If the user explicitly mentioned a movie title in their query (e.g., "Triple Cross"), start your response by acknowledging that movie, like: "For a movie similar to '[Movie Name from Query]', consider these:"
    2. Then, suggest 1 to 3 movies from the provided context that are similar or relevant.
    3. For each suggested movie, **extract its exact Title from the "Title: " prefix in the provided details.**
    4. Present each suggestion with its **Title first**, followed by a brief explanation of why it's a good suggestion, drawing from its plot summary.
    5. If no relevant suggestions can be made from the provided context, state that clearly.

    Question: {query}
    Answer:
    """
    return llm.predict(prompt)

def answer_user_query(query):
    qtype = classify_query_type(query)
    if qtype == "structured":
        return handle_structured_query(query)
    else:
        return handle_semantic_query(query)



# # --- Initialize Session ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
    
# # if "display_history" not in st.session_state:
# #     st.session_state.display_history = []

# # Use an index to track which conversation is currently displayed
# if "current_chat_index" not in st.session_state:
#     st.session_state.current_chat_index = -1  # -1 means show the full chat history

# # --- Sidebar for Chat History ---
# with st.sidebar:
#     st.header("Chat History")
#     if st.button("Clear History"):
#         st.session_state.chat_history = []
#         st.session_state.current_chat_index = -1
#         # st.session_state.display_history = []
#         st.rerun()
        
#     st.markdown("---")

#     # Add a button to show the full conversation
#     if st.button("Show All Conversations"):
#         st.session_state.current_chat_index = -1
#         st.rerun()
    
#     for i, chat in enumerate(st.session_state.chat_history):
#         if st.button(f"**{i+1}.** {chat['user'][:25]}...", key=f"hist_{i}"):
#             # Set the main display history to the selected chat and rerun
#             # st.session_state.display_history = [chat]
#             st.session_state.current_chat_index = i
#             st.rerun()


# # --- Main Chat Interface ---
# st.title("🎬 Movie RAG Assistant")

# # Logic to determine which chat history to display
# if st.session_state.current_chat_index == -1:
#     # Display the full history
#     display_chats = st.session_state.chat_history
# else:
#     # Display only the selected chat from the sidebar
#     display_chats = [st.session_state.chat_history[st.session_state.current_chat_index]]


# # Chat display
# for chat in st.session_state.chat_history:
#     st.markdown(f"<div class='stChatMessage user-msg'>You: {chat['user']}</div>", unsafe_allow_html=True)
#     if isinstance(chat['assistant'], pd.DataFrame):
#         # st.dataframe(chat['assistant'])
#         st.dataframe(chat['assistant'], use_container_width=True, height=400)
#     else:
#         st.markdown(f"<div class='stChatMessage assistant-msg'>Bot: {chat['assistant']}</div>", unsafe_allow_html=True)

# # User input
# user_query = st.text_input("Ask about movies...", placeholder="e.g., 'List movies by D.W. Griffith'")

# if st.button("Send"):
#     if user_query.strip():
#         with st.spinner("Thinking..."):
#             answer = answer_user_query(user_query)
#         # Append to the full history
#         st.session_state.chat_history.append({"user": user_query, "assistant": answer})
#         # Set the main display to the new conversation
#         # st.session_state.display_history = st.session_state.chat_history
#         st.session_state.current_chat_index = -1 # Go back to full history view
#         st.rerun()
# ----------------------------------------------------------------------

# --- Initialize Session ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "current_chat_index" not in st.session_state:
    st.session_state.current_chat_index = -1

# Callback function to set the current chat index
def set_chat_index(index):
    st.session_state.current_chat_index = index

# --- Sidebar for Chat History ---
with st.sidebar:
    st.header("Chat History")
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.session_state.current_chat_index = -1
        st.rerun()
        
    st.markdown("---")
    
    # Button to show the full conversation
    if st.button("Show All Conversations", on_click=set_chat_index, args=(-1,)):
        # This button is handled by the on_click callback
        pass
        
    for i, chat in enumerate(st.session_state.chat_history):
        # Use on_click to set the state
        st.button(
            f"**{i+1}.** {chat['user'][:25]}...",
            key=f"hist_{i}",
            on_click=set_chat_index,
            args=(i,)
        )

# --- Main Chat Interface ---
st.title("🎬 Movie RAG Assistant")

# Logic to determine which chat history to display
if st.session_state.current_chat_index == -1:
    # Display the full history
    display_chats = st.session_state.chat_history
else:
    # Display only the selected chat from the sidebar
    # Ensure the index is valid to avoid an IndexError
    if st.session_state.current_chat_index < len(st.session_state.chat_history):
        display_chats = [st.session_state.chat_history[st.session_state.current_chat_index]]
    else:
        # If the index is out of bounds (e.g., history was cleared), reset to full view
        st.session_state.current_chat_index = -1
        display_chats = st.session_state.chat_history


# Chat display loop now uses the `display_chats` list
for chat in display_chats:
    st.markdown(f"<div class='stChatMessage user-msg'>You: {chat['user']}</div>", unsafe_allow_html=True)
    if isinstance(chat['assistant'], pd.DataFrame):
        st.dataframe(chat['assistant'], use_container_width=True, height=400)
    else:
        st.markdown(f"<div class='stChatMessage assistant-msg'>Bot: {chat['assistant']}</div>", unsafe_allow_html=True)

# User input
user_query = st.text_input("Ask about movies...", placeholder="e.g., 'List movies by D.W. Griffith'")

if st.button("Send"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            answer = answer_user_query(user_query)
        st.session_state.chat_history.append({"user": user_query, "assistant": answer})
        st.session_state.current_chat_index = -1 # Go back to full history view
        st.rerun()