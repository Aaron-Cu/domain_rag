import os
import json
import nltk
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from nltk.tokenize import word_tokenize
from openai import OpenAI

nltk.download('punkt')

MODEL_ID = ''
DEFAULT_CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
RAG_CLIENT = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="lm-studio")

class Question(BaseModel):
    question: str

class QuestionsList(BaseModel):
    questions: List[Question]

class Answer(BaseModel):
    answer: str

questions_list_parser = PydanticOutputParser(pydantic_object=QuestionsList)
question_parser = PydanticOutputParser(pydantic_object=Question)
answer_parser = PydanticOutputParser(pydantic_object=Answer)
json_parser = JsonOutputParser()

question_instructions = question_parser.get_format_instructions()

few_shot_examples_questions = [
    {"role": "system", "content": "You generate relevant, contextual questions based on a given passage. The questions are scenario-based and phrased casually."},
    {"role": "user", "content": "Generate casual, practical questions someone might ask after reading a passage about a topic. Return only JSON with no extra text."},
    {"role": "assistant", "content": json.dumps([
        {"question": "What does this mean in simpler terms?"},
        {"question": "How would I apply this in real life?"},
        {"question": "What are some examples of this in action?"}
    ])}
]

few_shot_examples_answers = [
    {"role": "user", "content": "Why is this concept important?"},
    {"role": "assistant", "content": "This concept is important because it helps with understanding broader systems and decision-making in context. For example..."}
]

def json_to_dataframe(json_input: str) -> pd.DataFrame:
    try:
        with open(json_input, 'r') as json_file:
            data_list = json.load(json_file)
        return pd.DataFrame(data_list)
    except Exception:
        return pd.DataFrame()

def parse_with_fixer(response_content: str, parser) -> Tuple[bool, any]:
    try:
        return True, parser.parse(response_content)
    except Exception:
        try:
            fixed_output = json.loads(response_content)
            return True, parser.parse_obj(fixed_output)
        except Exception:
            return False, None

def add_to_data_list(data: list, question: str, answer: str, head: str, text: str):
    data.append({'question': question, 'answer': answer, 'head': head, 'text': text})

def parse_directory(directory: str) -> pd.DataFrame:
    data = []
    for file in Path(directory).glob("*.json"):
        df = json_to_dataframe(file)
        if not df.empty:
            data.append(df)
    if data:
        return pd.concat(data, ignore_index=True)
    raise ValueError("No valid JSON files found.")

def generate_questions(client, context: str, num_questions: int = 3) -> List[str]:
    prompt = PromptTemplate(
        template="""Given the following context, generate {num_questions} scenario-based questions someone might ask to better understand or apply the information. The questions should be casually phrased and practical.
            Context: {context}
            {format_instructions}
            Responses should be in JSON format only.""",
        input_variables=["context", "num_questions"],
        partial_variables={"format_instructions": question_instructions}
    )
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=few_shot_examples_questions + [
            {"role": "user", "content": prompt.format(context=context, num_questions=num_questions)}
        ],
        temperature=1.7,
    )
    content = completion.choices[0].message.content
    valid, parsed = parse_with_fixer(content, json_parser)
    if valid:
        return [q['question'] for q in parsed]
    raise ValueError("Invalid question format.")

def generate_answer(client, question: str) -> str:
    prompt = PromptTemplate(template="{question}", input_variables=["question"])
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=few_shot_examples_answers + [{"role": "user", "content": prompt.format(question=question)}],
        temperature=0.7,
    )
    return completion.choices[0].message.content

def count_tokens(text: str) -> int:
    return len(word_tokenize(text))

def process_data(
    df: pd.DataFrame,
    client=DEFAULT_CLIENT,
    answer_client=RAG_CLIENT,
    num_questions: int = 3,
    num_retries: int = 3,
) -> pd.DataFrame:
    data = []

    def process_single_context(context_row):
        context_text = context_row.get('Chunk', '')
        head_title = context_row.get('Title', '')

        try:
            questions = generate_questions(client, context_text, num_questions=num_questions)
        except Exception:
            return

        for q in questions:
            for _ in range(num_retries):
                try:
                    answer = generate_answer(answer_client, q)
                    add_to_data_list(data, q, answer, head_title, context_text)
                    break
                except Exception:
                    continue

    for i in tqdm(range(len(df)), desc="Processing contexts"):
        process_single_context(df.iloc[i])

    result_df = pd.DataFrame(data)
    return result_df

def process_all(directory: str) -> pd.DataFrame:
    if not Path(directory).exists():
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    df = parse_directory(directory)
    return process_data(df)