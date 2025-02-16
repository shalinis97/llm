import os
import json
import requests
import logging
import subprocess
import sqlite3
import datetime
import re
import pytesseract
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter
import speech_recognition as sr
import markdown
import git
from bs4 import BeautifulSoup
import numpy as np
import datetime
import re
from dateutil import parser
import shutil


# To Do
# Vectorize functions to reduce LLM usage
# Put all the functions in a separate file

# Common date formats
# Expanded list of common date formats (with and without time)
DATE_FORMATS = [
    "%Y-%m-%d",        # 2024-02-13
    "%Y/%m/%d",        # 2022/09/01
    "%d-%b-%Y",        # 05-Jan-2009
    "%d/%m/%Y",        # 13/02/2024
    "%m/%d/%Y",        # 02/13/2024
    "%d %b, %Y",       # 11 Sep, 2007
    "%b %d, %Y",       # Sep 11, 2007
    "%d %B %Y",        # 11 September 2007
    "%B %d, %Y",       # September 11, 2007
    "%d %b %Y",        # 11 Sep 2007 (without comma)
    "%b %d %Y",        # Sep 11 2007 (without comma)
    "%Y-%m-%d %H:%M:%S",  # 2022-09-01 17:35:04
    "%Y/%m/%d %H:%M:%S"   # 2022/09/01 17:35:04
]

# AI Proxy URLs
AI_PROXY_CHAT_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AI_PROXY_EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Security setting
SECURE_DATA_PATH = "/data/"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security functions
def enforce_data_security(file_path):
    logger.info(f"Checking file path: {file_path}")
    if not file_path.startswith(SECURE_DATA_PATH):
        raise PermissionError(f"Access denied: {file_path}")

def file_folder_deletion(file_path):
    logger.info(f"Checking file path: {file_path}")
    raise PermissionError(f"Deletion not allowed: {file_path}")

# AI Proxy LLM Call with Function Calling
def ask_llm(task_description):
    """Calls LLM and extracts structured function calls from JSON output."""
    
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}

    function_definitions = [
        {
            "name": "install_package",
            "description": "Install a Python package",
            "parameters": {
                "type": "object",
                "properties": {"package": {"type": "string"}},
                "required": ["package"]
            }
        },
        {
            "name": "execute_script",
            "description": "Run a Python script",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["script"]
            }
        },
        {
            "name": "write_file",
            "description": "Writes to a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]
            }
        },
        {
            "name": "read_file",
            "description": "Reads a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        },
        {
            "name": "count_weekdays",
            "description": "Count weekdays in a file and write the count to file",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "weekday": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["file", "weekday", "output_file"]
            }
        },
        {
            "name": "extract_email",
            "description": "Extract email from a file and writing to another file",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["file", "output_file"]
            }
        },
        {
            "name": "extract_credit_card",
            "description": "Extract credit card number from a file and writing to another file",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["file", "output_file"]
            }
        },
        {
            "name": "get_recent_log_lines",
            "description": "Get recent log lines from files in a directory and write them to a file" ,
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["directory", "output_file"]
            }
        },
        {
            "name": "execute_sql",
            "description": "Execute given SQL query on a SQLite database and write output to a file",
            "parameters": {
                "type": "object",
                "properties": {"db_path": {"type": "string"}, "query": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["db_path", "query"]
            }
        },
        {
            "name": "fetch_api_data",
            "description": "Fetch data from an API and write it to a file",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["url", "output_file"]
            }
        },
        {
            "name": "clone_git_repo",
            "description": "Clone a Github repo and commit changes",
            "parameters": {
                "type": "object",
                "properties": {"repo_url": {"type": "string"}, "branch": {"type": "string"}, "commit_message": {"type": "string"}},
                "required": ["repo_url"]
            }
        },
        {
            "name": "scrape_website",
            "description": "Scrape a website and write the content to a file",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["url", "output_file"]
            }
        },
        {
            "name": "transcribe_audio",
            "description": "Transcribe audio file to text",
            "parameters": {
                "type": "object",
                "properties": {"input_file": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["input_file", "output_file"]
            }
        },
        {
            "name": "convert_markdown_to_html",
            "description": "Convert Markdown to HTML",
            "parameters": {
                "type": "object",
                "properties": {"input_file": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["input_file", "output_file"]
            }
        },
        {
            "name": "sort_json",
            "description": "Sort JSON file by keys and write to output file",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"},"output_file": {"type": "string"}, "keys": {"type": "string"}},
                "required": ["file"]
            }
        },
        {
            "name": "extract_markdown_headers",
            "description": "Extract headers from Markdown files in a directory and writes to index.json",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["directory"]
            }
        },
        {
            "name": "find_similar_comments",
            "description": "Find the most similar pair of comments in a file",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "output_file": {"type": "string"}},
                "required": ["file","output_file"]
            }
        },
        {
            "name": "format_markdown",
            "description": "Format a Markdown file using Prettier@3.4.2",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}},
                "required": ["file"]
            }
        },
        {
            "name": "install_uv_and_run_script",
            "description": "Install uv and run a Python script",
            "parameters": {
                "type": "object",
                "properties": {"script_url": {"type": "string"}, "user_email": {"type": "string"}},
                "required": ["script_url", "user_email"]
            }
        },
        {  
            "name": "execute_linux_command",
            "description": "Execute a Linux command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        },
        {
            "name" : "compress_or_resize_image",
            "description": "Compress or resize an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "output_file": {"type": "string"},
                    "quality": {"type": "number"},
                    "max_width": {"type": "number"},
                    "max_height": {"type": "number"}
                },
                "required": ["file", "output_file"]
            }
        },
        {
            "name": "filter_csv",
            "description": "Filter a CSV file based on a column value and save it as JSON",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "filter_column": {"type": "string"},
                    "filter_value": {"type": "string"},
                    "output_file": {"type": "string"}
                },
                "required": ["file", "filter_column", "filter_value", "output_file"]
            }
        },
        {
            "name": "file_folder_deletion",
            "description": "Delete a file or folder",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        }  
    ]

    system_message = {
        "role": "system",
        "content": (
            "You are an automation assistant. Convert the given task into structured function calls."
            "Respond in JSON format with a list of function calls, where each function call includes:\n"
            "- `name`: The function to execute\n"
            "- `arguments`: JSON object containing parameter values\n\n"
            "Example:\n"
            "```json\n"
            "[\n"
            "    { \"name\": \"format_markdown\", \"arguments\": { \"package\": \"numpy\" } },\n"
            "    { \"name\": \"execute_linux_command\", \"arguments\": { \"script\": \"/data/script.py\", \"args\": [\"--verbose\"] } }\n"
            "]\n"
            "```"
        )
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            system_message,
            {"role": "user", "content": task_description}
        ],
        "functions": function_definitions,
        "temperature": 0
    }
    logger.info(f"Before LLM API Call:")
    response = requests.post(AI_PROXY_CHAT_URL, json=payload, headers=headers)
    logger.info(f"After LLM API Call:")

    if response.status_code == 200:
        try:
            data = response.json()
            logger.info(f"LLM Response data: {data}")
            choices = data.get("choices", [])
            logger.info(f"LLM Response choices: {choices}")

            if not choices:
                logger.error("No choices returned by LLM.")
                return []

            content = choices[0].get("message", {}).get("content", "").strip()

            # Extract JSON from content
            if content.startswith("```json"):
                content = content[7:-3].strip()  # Remove ```json and ``` markers
            
            steps = json.loads(content)  # Parse content into a list of function calls
            
            logger.info(f"Extracted LLM Function Calls: {steps}")
            return steps
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM JSON content: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing LLM response: {e}")
            return []
    else:
        logger.error(f"LLM API Error: {response.text}")
        return []

# Utility functions
def read_file(path):
    enforce_data_security(path)
    logger.info(f"Reading the file {path}")
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"File {path} not found."
    return f"File {path} read successfully."

def write_file(path, content):
    enforce_data_security(path)
    logger.info(f"Writing to file {path}")
    with open(path, "w") as file:
        file.write(content)
    return f"File {path} written successfully."

def install_uv_and_run_script(script_url, user_email):
    """Installs `uv` (if required), downloads `datagen.py`, and runs it with `user.email`."""
    logger.info(f"Installing uv and running script {script_url} with email {user_email}")

    uv_check = shutil.which("uv")

    if uv_check:
        logger.info(f"uv is already installed")
        # uv_check = subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # logger.info(f"uv_check {uv_check}")
    # # Step 1: Check if `uv` is installed
    # try:
    #     logger.info(f"Before subprocess.run uv --version")
    #     subprocess.run(["uv", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     logger.info(f"After subprocess.run uv --version")
    #     uv_installed = True
    # except subprocess.CalledProcessError as e:
    #     logger.info(f"uv is not installed {str(e)}")
    #     pass
    # #    uv_installed = False

    
    # Step 2: Install `uv` if not installed
    if not uv_check:
        try:
            logger.info("Installing uv...")
            subprocess.run(["pip", "install", "uv"], check=True)
        except subprocess.CalledProcessError as e:
            return f"Error installing uv: {str(e)}"

    # Step 3: Download `datagen.py` if not already present
    script_path = "datagen.py"
     
    if not os.path.exists(script_path):
        logger.info("Downloading script...")
        try:
            response = requests.get(script_url, timeout=15)
            if response.status_code == 200:
                with open(script_path, "wb") as f:
                    f.write(response.content)
            else:
                return f"Failed to download script. HTTP Status: {response.status_code}"
        except requests.RequestException as e:
            return f"Error downloading script: {str(e)}"

    # Step 4: Run the script with `user.email`
    try:
        logger.info(f"Running script... {script_path} with email {user_email}")
        subprocess.run(["uv", "run", script_path, user_email], check=True)
        os.remove(script_path)
        return f"Script {script_path} executed successfully with email {user_email} and removed the script."
    except subprocess.CalledProcessError as e:
        return f"Error executing script: {str(e)}"    

def install_package(package):
    logger.info(f"Installing package {package}")
    subprocess.run(["pip", "install", package], check=True)
    return f"Package {package} installed successfully."

def execute_linux_command(command):
    logger.info(f"Executing linux command {command}")
    subprocess.run([command], check=True)
    return f"Command {command} executed successfully."

def execute_script(script, args=[]):
    enforce_data_security(script)
    logger.info(f"Executing script {script} with args {args}")
    subprocess.run(["python3", script] + args, check=True)
    return f"Script {script} executed successfully."

def clean_date_string(date_str):
    """Cleans the date string by removing extra spaces, special characters, and time if present."""
    # logger.info(f"Cleaning date string: {date_str}")
    date_str = date_str.strip()  # Remove leading/trailing spaces
    date_str = re.sub(r"[^\w\s/-]", "", date_str)  # Remove special characters except / and -
    date_str = re.split(r"\s+", date_str)[0]  # Remove time component if present
    return date_str

def parse_date(date_str):
    """Tries multiple formats to parse a date string after cleaning, with a fallback to dateutil.parser."""
    cleaned_date = clean_date_string(date_str)
    logger.info(f"Cleaned date string: {cleaned_date}")

    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(cleaned_date, fmt)
        except ValueError:
            continue

    # **Fallback: Use dateutil.parser if standard formats fail**
    try:
        return parser.parse(date_str, fuzzy=True)
    except ValueError:
        raise ValueError(f"Unknown date format: {date_str}")

def count_weekdays(file, weekday, output_file=None):
    """Counts occurrences of a specific weekday in a file and optionally writes to a file."""
    enforce_data_security(file)
    if output_file:
        enforce_data_security(output_file)

    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    unknown_dates = []

    try:
        with open(file, "r") as f:
            dates = [line.strip() for line in f.readlines() if line.strip()]

        count = 0
        for d in dates:
            try:
                parsed_date = parse_date(d)
                if parsed_date.weekday() == weekday_map[weekday]:
                    count += 1
            except ValueError:
                unknown_dates.append(d)

        # Log unknown date formats
        if unknown_dates:
            logger.warning(f"Skipped {len(unknown_dates)} lines due to unknown date formats: {unknown_dates}")

        if output_file:
            with open(output_file, "w") as f:
                f.write(str(count))
            return f"Result written to {output_file}."
        else:
            return f"{weekday} count: {count}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def extract_email(file, output_file):
    enforce_data_security(file)
    logger.info(f"Extracting email from file {file} and writing to {output_file}")
    with open(file, "r") as f:
        text = f.read()
    match = re.search(r"From:.*<([^>]+)>", text)
    if match:
        with open(output_file, "w") as f:
            f.write(match.group(1))
        return f"Email extracted and saved to {output_file}."

def preprocess_image(file):
    """Preprocess the image to improve OCR accuracy."""
    logger.info(f"Preprocessing image {file}")
    image = Image.open(file).convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
    logger.info(f"Enhancing Image")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(5)  # Increase contrast
    # image = image.point(lambda p: 0 if p < 100 else 255)  # Apply thresholding
    return image

def extract_credit_card(file, output_file):
    """Extracts a credit card number from an image using OCR and writes it to a file."""
    enforce_data_security(file)
    enforce_data_security(output_file)

    try:
        # Preprocess image
        image = preprocess_image(file)

        # OCR Configuration: Only allow digits, better segmentation
        config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"

        logger.info(f"Before OCR image_to_string")
        text = pytesseract.image_to_string(image, config=config)

        # Extract credit card number using regex
        match = re.search(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", text)
        
        if match:
            card_number = match.group(0).replace(" ", "").replace("-", "")  # Normalize number
            with open(output_file, "w") as f:
                f.write(card_number)
            return f"Credit card number extracted and saved to {output_file}."
        else:
            return "No valid credit card number found in the image."
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_recent_log_lines(directory, output_file):
    enforce_data_security(directory)
    enforce_data_security(output_file)
    logger.info(f"get_recent_log_lines from directory {directory} and writing to {output_file}")
    log_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".log")], key=os.path.getmtime, reverse=True)
    with open(output_file, "w") as f:
        for log in log_files[:10]:
            with open(log, "r") as lf:
                f.write(lf.readline().strip() + "\n")

    return f"Last 10 log lines written to {output_file}"

def sort_json(file, output_file=None, keys=None):
    """Sorts contacts from a JSON file in ascending order by last_name, then first_name, and writes only JSON output."""
    enforce_data_security(file)
    enforce_data_security(output_file)
    logger.info(f"Sorting JSON file {file} and writing to {output_file}")

    try:
        with open(file, "r", encoding="utf-8") as f:
            contacts = json.load(f)

        if not isinstance(contacts, list):
            return "Error: JSON file must contain a list of contacts."

        # Default sorting keys if none provided
        if not keys:
            keys = ["last_name", "first_name"]

        # Sort contacts in ascending order
        sorted_contacts = sorted(
            contacts, 
            key=lambda x: (
                (x.get("last_name") or "").strip().lower(), 
                (x.get("first_name") or "").strip().lower()
            )
        )

        if not output_file:
            output_file = file  # Overwrite the input file if no output file is provided

        # Write only JSON output to the file (no extra text)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sorted_contacts, f, separators=(",", ":"))

        return f"Sorted contacts written to {output_file}."  # No extra return message, just pure JSON output in the file
    except Exception as e:
        return f"Error processing JSON file: {str(e)}"


def extract_markdown_headers(directory, output_file="/data/docs/index.json"):
    """Finds all Markdown (.md) files in a directory, extracts the first H1 header, and writes a single-line JSON list to a file."""
    enforce_data_security(directory)
    enforce_data_security(output_file)
    logger.info(f"Extracting Markdown headers from {directory} and writing to {output_file}")

    index = {}

    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            match = re.match(r"^#\s+(.+)", line.strip())  # Match first H1
                            if match:
                                relative_path = os.path.relpath(file_path, directory)  # Get relative path
                                index[relative_path] = match.group(1)
                                break  # Stop after first H1

        # Write the JSON list as a single line
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(index, f, separators=(",", ":"))  # Compact single-line format

        return f"Markdown index written as a single-line JSON to {output_file}."
    except Exception as e:
        return f"Error processing Markdown files: {str(e)}"

def execute_sql(db_path, query, output_file="/data/sql-output.txt"):
    enforce_data_security(db_path)
    enforce_data_security(output_file)
    logger.info(f"Executing SQL query {query} on database {db_path} and writing to {output_file}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchone()[0]
    conn.close()
    result = result if result is not None else 0
    with open(output_file, "w") as f:
            f.write(str(result))
    return f"Result written to {output_file}."

def fetch_api_data(url, output_file):
    enforce_data_security(output_file)
    logger.info(f"Fetching API data from {url} and writing to {output_file}")
    response = requests.get(url)
    write_file(output_file, response.text)
    return f"API data written to {output_file}"

def clone_git_repo(repo_url, branch="main", commit_message="Dummy commit"):
    logger.info(f"Cloning git repo {repo_url} with branch {branch} and commit message {commit_message}")
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    git.Repo.clone_from(repo_url, repo_name)
    repo = git.Repo(repo_name)
    repo.git.checkout(branch)
    with open(f"{repo_name}/dummy.txt", "w") as f:
        f.write("Auto-commit")
    repo.git.add(A=True)
    repo.git.commit("-m", commit_message)
    repo.git.push()
    return f"Cloned repo {repo_url} with branch {branch} and committed {commit_message} and pushed"

def scrape_website(url, output_file):
    enforce_data_security(output_file)
    logger.info(f"Scraping website {url} and writing to {output_file}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    write_file(output_file, soup.prettify())
    return f"Scraped content written to {output_file}"

def transcribe_audio(input_file, output_file):
    enforce_data_security(output_file)
    enforce_data_security(input_file)
    logger.info(f"Transcribing audio from {input_file} and writing to {output_file}")
    recognizer = sr.Recognizer()
    with sr.AudioFile(input_file) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    write_file(output_file, text)
    return f"Transcribed audio written to {output_file}"

def convert_markdown_to_html(input_file, output_file):
    enforce_data_security(output_file)
    enforce_data_security(input_file)
    logger.info(f"Converting markdown to HTML from {input_file} and writing to {output_file}")
    with open(input_file, "r") as f:
        html = markdown.markdown(f.read())
    write_file(output_file, html)
    return f"Markdown converted to HTML written to {output_file}"

def find_similar_comments(file, output_file):
    logger.info(f"Finding similar comments in {file} and writing to {output_file}")
    """Finds the most similar pair of comments using OpenAI embeddings."""
    enforce_data_security(file)
    enforce_data_security(output_file)

    with open(file, "r") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    if len(comments) < 2:
        with open(output_file, "w") as f:
            f.write("Not enough comments to compare.")
        return "Not enough comments to compare."

    # Request embeddings in a batch for efficiency
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": comments}
    
    response = requests.post(AI_PROXY_EMBEDDING_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch embeddings: {response.text}")

    embeddings = response.json().get("data", [])
    if len(embeddings) != len(comments):
        raise ValueError("Mismatch between comments and embeddings count.")

    # Compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    max_similarity = -1
    best_pair = ("", "")

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            similarity = cosine_similarity(embeddings[i]["embedding"], embeddings[j]["embedding"])
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (comments[i], comments[j])

    with open(output_file, "w") as f:
        f.write(f"{best_pair[0]}\n{best_pair[1]}")

    return f"Most similar comments written to {output_file}"

def format_markdown(file):
    logger.info(f"Formatting markdown file {file}")
    """Formats a Markdown file using Prettier@3.4.2."""
    enforce_data_security(file)

    try:
        subprocess.run(["npx", "prettier@3.4.2", "--write", file])
        return f"Formatted {file} successfully with Prettier@3.4.2."
    except subprocess.CalledProcessError as e:
        return f"Error formatting {file}: {str(e)}"

def compress_or_resize_image(file, output_file, quality=80, max_width=None, max_height=None):
    logger.info(f"Compressing or resizing image {file} to {output_file}")
    """Compresses or resizes an image based on given constraints."""
    enforce_data_security(file)
    enforce_data_security(output_file)

    try:
        img = Image.open(file)

        # Resize if max width or height is specified
        if max_width or max_height:
            img.thumbnail((max_width or img.width, max_height or img.height))

        # Save compressed image
        img.save(output_file, quality=quality, optimize=True)
        return f"Image saved to {output_file} with quality {quality}."
    except Exception as e:
        return f"Error processing image: {str(e)}"

def filter_csv(file, filter_column, filter_value, output_file):
    """Filters a CSV file based on a column value and saves it as JSON."""
    enforce_data_security(file)
    enforce_data_security(output_file)
    logger.info(f"Filtering CSV file {file} based on column {filter_column} and value {filter_value} and writing to {output_file}")

    try:
        with open(file, "r") as f:
            reader = csv.DictReader(f)
            filtered_data = [row for row in reader if row.get(filter_column) == filter_value]

        with open(output_file, "w") as f:
            json.dump(filtered_data, f, indent=2)

        return f"Filtered data saved to {output_file}."
    except Exception as e:
        return f"Error processing CSV: {str(e)}"
        
# Function dispatcher
function_map = {
    "install_package": install_package,
    "read_file": read_file,
    "write_file": write_file,
    "execute_script": execute_script,
    "count_weekdays": count_weekdays,
    "extract_email": extract_email,
    "extract_credit_card": extract_credit_card,
    "get_recent_log_lines": get_recent_log_lines,
    "sort_json": sort_json,
    "extract_markdown_headers": extract_markdown_headers,
    "execute_sql": execute_sql,
    "fetch_api_data": fetch_api_data,
    "clone_git_repo": clone_git_repo,
    "scrape_website": scrape_website,
    "transcribe_audio": transcribe_audio,
    "convert_markdown_to_html": convert_markdown_to_html,
    "find_similar_comments": find_similar_comments,
    "format_markdown": format_markdown,
    "execute_linux_command": execute_linux_command,
    "install_uv_and_run_script": install_uv_and_run_script,
    "compress_or_resize_image": compress_or_resize_image,
    "filter_csv": filter_csv,
    "file_folder_deletion": file_folder_deletion
}

# Task Runner
def task_runner(task_description):
    structured_steps = ask_llm(task_description)
    output = []
    logger.info(f"structured_steps {structured_steps}")
    if not isinstance(structured_steps, list):
        structured_steps = [structured_steps]

    for step in structured_steps:
        action = step.get("name")
        params = step.get("arguments", {})

        if action in function_map:
            try:
                result = function_map[action](**params)
                output.append(f"{action} executed successfully: {result}")
            except Exception as e:
                logger.error(f"Error executing {action}: {e}")
                output.append(f"Error in {action}: {str(e)}")
        else:
            output.append(f"Unknown action: {action}")
    return "\n".join(output)

# API Endpoints
@app.post("/run")
async def run_task(task: str = Query(..., description="Task description")):
    try:
        result = task_runner(task)
        return {"status": "success", "output": result}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/read")
def read_file(path: str):
    """Reads the content of the specified file and returns it correctly formatted."""
    enforce_data_security(path)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        # Detect if it's a JSON file
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)  # Load JSON properly
            return content  # ✅ Returns proper JSON (not a quoted string)

        # Return plain text for other file types
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content, media_type="text/plain")   # ✅ Returns raw text (not double-quoted JSON string)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format in file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")