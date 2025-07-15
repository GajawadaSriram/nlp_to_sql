from flask import Flask, render_template, request, redirect, url_for, session
import subprocess
import threading
import sqlite3
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import faiss
import numpy as np
import re
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from io import StringIO
import csv
from flask import send_file
app = Flask(__name__)
app.secret_key = 'your_secret_key' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoint\checkpoint-2500"
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = T5Tokenizer.from_pretrained("T5-Checkpoints\T5-Checkpoints")



# os.environ["GROQ_API_KEY"] = groq_api_key
load_dotenv()
chat_groq = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model="llama-3.3-70b-versatile")



def create_schema_embeddings(schema_info):
    schema_texts = []
    for table, columns in schema_info.items():
        table_description = f"Table {table} with columns: {', '.join(columns)}"
        schema_texts.append(table_description) 
        for col in columns:
            schema_texts.append(f"{table}.{col}")


    dimension = 512
    index = faiss.IndexFlatL2(dimension)
    schema_vectors = np.array([get_embedding(text).flatten() for text in schema_texts])
    index.add(schema_vectors)
    return index, schema_texts


default_schema_info = {
    "students": ["student_id", "name", "score"],
    "classes": ["class_id", "class_name", "teacher_id"],
    "teachers": ["teacher_id", "teacher_name", "subject"]
}

def generate_sql(question):
    input_text = f"SQL generation: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    human_readable_sql = re.search(r"'human_readable':\s'(.*)',", sql_query)
    
    if human_readable_sql:
        return human_readable_sql.group(1)  
    else:
        return sql_query  

def sanitize_string(text):
    return text.replace("\\", "")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.encoder(**inputs).last_hidden_state.mean(dim=1)
    embedding = outputs.detach().cpu().numpy()
    return embedding

def get_relevant_schema(query, index, schema_texts, top_k=10):
    query_vector = get_embedding(query)
    _, indices = index.search(query_vector, top_k)

    relevant_elements = [schema_texts[i] for i in indices[0]]
    tables = set([elem.split('.')[0] for elem in relevant_elements if '.' in elem])
    
    expanded_elements = [elem for elem in schema_texts if any(table in elem for table in tables)]
    return list(set(relevant_elements + expanded_elements))


def generate_schema_corrected_sql(question, rough_sql, schema_elements):
    schema_independent_keywords = ['create', 'drop', 'alter', 'truncate', 'rename', 'delete']
    # print(question)
    # print(rough_sql)
    # print(schema_elements)
    if any(keyword in question.lower() for keyword in schema_independent_keywords):
        prompt = f"""
        For the question: "{question}", and the rough SQL: "{rough_sql}", convert the following rough SQL query into a valid SQL query.
        
        ### striclty follow the table name that is in the rough sql and question not the one in schema
        ### ONLY if the table name in the question and in the schema matches exactly else return invalid table name for DML commands.
        ### If the question says "delete the table", then use DROP TABLE.
        ### If the question says "delete rows from table" or just "delete from", then use DELETE FROM
        ### Give the proper and related error description for the error
        ### ONLY return the correct SQL query. NO explanations, NO formatting, and NO code blocks.
        ### Just output the SQL statement, nothing else.
        """
    else:
        table_columns = {}
        for elem in schema_elements:
            if '.' in elem:
                table, col = elem.split('.')
                table_columns.setdefault(table, []).append(col)

        schema_string = ""
        for table, columns in table_columns.items():
            schema_string += f"\nTable {table}: columns {', '.join(columns)}."

        prompt = f"""
        Given the following database schema: {schema_string}

        For the question: "{question}", convert the following rough SQL query into a valid SQL query that matches the schema correctly.

        Rough SQL: {rough_sql}
        ### striclty follow the table name that is in the rough sql and question not the one in schema
        ### ONLY if the table name in the question and in the schema matches exactly else return invalid table name for DML commands.
        ### Give the proper and related error description for the error like if deleting the table that is not exisiting give there is  no table name .Give the correct description
        ### ONLY return the correct SQL query. NO explanations, NO formatting, and NO code blocks.
        ### Just output the SQL statement, nothing else.
        """

    prompt_template = PromptTemplate(template=prompt, input_variables=["question", "rough_sql"])
    chain = prompt_template | chat_groq

    try:
        corrected_sql = chain.invoke({"rough_sql": rough_sql})
        print(corrected_sql)
        return corrected_sql
    except Exception as e:
        return f"Error: {str(e)}"


def get_schema_from_db(db_path):

    schema_info = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sample_table';")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()  
    # print(tables)
    for table_tuple in tables:
        table_name = table_tuple[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        
        columns_info = cursor.fetchall()
        
        column_names = [column[1] for column in columns_info]
        schema_info[table_name] = column_names
    
    conn.close()
    return schema_info

def process_nlp_to_sql(db_path, question):

    schema_info = get_schema_from_db(db_path)

    index, schema_texts = create_schema_embeddings(schema_info)

    generated_sql = generate_sql(question)

    schema_elements = get_relevant_schema(question, index, schema_texts)

    refined_sql = generate_schema_corrected_sql(question, generated_sql, schema_elements)

    return sanitize_string(refined_sql.content)

def refresh_schema_cache(db_path):
    schema_info = get_schema_from_db(db_path)
    index, schema_texts = create_schema_embeddings(schema_info)
    app.schema_cache[db_path] = (schema_info, index, schema_texts)

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def init_user_data_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sample_table (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER
        )
    ''')
    conn.commit()
    conn.close()


@app.route('/')
def home():
    return redirect(url_for('index'))
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[0], password): 
            session['username'] = username 
            return redirect(url_for('dashboard'))
            
              
        else:
            return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html')  

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return redirect(url_for('dashboard'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('dashboard'))

    if file and file.filename.endswith('.csv'):
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # Connect to the database
            conn = sqlite3.connect('user_data.db')
            cursor = conn.cursor()

            # Create a table based on the CSV file name (you can customize this)
            table_name = os.path.splitext(file.filename)[0]
            df.to_sql(table_name, conn, if_exists='replace', index=False)

            conn.commit()
            conn.close()

            return redirect(url_for('dashboard', success_message='CSV file uploaded and data inserted successfully!'))

        except Exception as e:
            return redirect(url_for('dashboard', error=f'Error uploading CSV: {str(e)}'))

    return redirect(url_for('dashboard', error='Invalid file format. Please upload a CSV file.'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password) 
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                           (username, hashed_password))
            conn.commit()
            return redirect(url_for('login'))  
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists.')
        finally:
            conn.close()

    return render_template('register.html') 


@app.route('/logout')
def logout():
    session.pop('username', None) 
    return redirect(url_for('login'))  




@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    db_path = 'user_data.db'
    question = ""
    output = None
    error = None
    success_message = None
    generated_sql = None
    query_history = session.get('query_history', [])

    if request.method == 'POST':
        question = request.form['sql_command'].strip()

        if not question:
            error = "Please enter a valid SQL query or natural language question."
        else:
            try:
                # Convert NLP to SQL
                generated_sql = process_nlp_to_sql(db_path, question)
                print("Generated SQL:",generate_sql)
                # Validate table/column existence (based on your logic)
                if "Invalid table or column" in generated_sql:
                    error = "Your question references a table or column that doesn't exist in the database schema."
                    return render_template('dashboard.html', schema_info=get_schema_from_db(db_path),
                                           sql_command=question, output=None, error=error,
                                           success_message=None, query_history=query_history)

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(generated_sql)
                conn.commit()

                if generated_sql.strip().lower().startswith('select'):
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    output = {'rows': rows, 'columns': columns}

                    # Store to query history
                    query_history.append({'question': question, 'sql': generated_sql})
                    session['query_history'] = query_history
                else:
                    success_message = "Query executed successfully!"

                conn.close()

            except sqlite3.OperationalError as e:
                if "already exists" in str(e).lower():
                    error = "The table you are trying to create already exists."
                else:
                    error = f"SQLite error: {str(e)}"
            except Exception as e:
                error = "Unable to process your input. Please reframe the question."

    # Always re-fetch schema at the end to reflect changes
    schema_info = get_schema_from_db(db_path)
    print("Latest schema from DB:", schema_info)
    return render_template('dashboard.html',
                           schema_info=schema_info,
                           sql_command=question,
                           output=output,
                           error=error,
                           success_message=success_message,
                           query_history=query_history)


init_user_data_db()
if __name__ == '__main__':
    init_db()  
    app.run(debug=False)
