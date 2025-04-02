# Ensure NLTK resources are available
nltk.download('punkt')

# Initialize OpenAI client
model='hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF'
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
RAG = OpenAI(base_url="http://127.0.0.1:5000", api_key="lm-studio")

class Question(BaseModel):
    question: str

class QuestionsList(BaseModel):
    questions: List[Question]

# Initialize the output parser for a list of questions
questions_list_parser = PydanticOutputParser(pydantic_object=QuestionsList)
questions_list_instructions = questions_list_parser.get_format_instructions()

class Answer(BaseModel):
    answer: str

question_parser = PydanticOutputParser(pydantic_object=Question)
answer_parser = PydanticOutputParser(pydantic_object=Answer)

question_instructions = question_parser.get_format_instructions()
answer_instructions = answer_parser.get_format_instructions()

# Initialize the JSON parser for fixing outputs
json_parser = JsonOutputParser()

few_shot_examples_questions = [
    {"role": "system", "content": "Taking on the persona of a family member, you generate relevant questions that caregivers dealing with specific medical conditions would ask. The questions you generate are scenario-based, and use informal and simple wording in the form of a casual conversation."},
    {"role": "user", "content": "Generate questions that a caregiver might ask while caring for someone with dementia. Consider their emotional and physical needs. Responses should be in JSON format with no other text or attributes."},
    {"role": "assistant", "content": json.dumps([
        {"question": "What are some things I can do to make bath time more enjoyable for my husband with dementia?"},
        {"question": "How can I encourage my mom with dementia to drink enough fluids?"},
        {"question": "What strategies can I use to reduce stress and anxiety when caring for a loved one with dementia?"},
        {"question": "How can I make mealtime easier for my dad with dementia?"},
        {"question": "What are some ways to prevent agitation during bath time for my mom with dementia?"}
    ])}
]

few_shot_examples_best_question = [
    {"role": "system", "content": "As a caregiver, you need to address the underlying needs driving your loved one's behavior. What are some specific reasons that might be contributing to this behavior? Provide at least two possible explanations and suggest ways to respond to each."},

    {"role": "user", "content": json.dumps({"question": "How can I assess which specific needs my person with dementia is trying to communicate through their behavior?"})},

    {"role": "assistant", "content":json.dumps([{"question": "What are some strategies to help identify what they're trying to communicate through their behavior? Provide at least two possible explanations and suggest ways to respond to each."}])}
]

few_shot_examples_answers = [
    {"role": "user", "content": "My mom with dementia keeps trying to wander outside. What can I do to keep her safe?"},
    {"role": "assistant", "content": "Consider the underlying reasons for wandering, such as looking for something familiar or seeking independence. Some strategies include: Designate a Safe Space: Create an area in your home where your mom feels comfortable and secure. Identify Triggers: Reflect on what might be causing her wanderings, such as anxiety or disorientation. Offer Choices: Provide options for activities or movements that she can control, like choosing to move around the room or engage in a favorite task. Use Non-Physical Barriers: Utilize physical obstacles to limit access to areas she's prone to wandering into, such as installing locks on doors or using gates at stairs. Redirect Attention: Engage your mom in activities or conversations that are calming and stimulating. Monitor Progress: Track changes in her behavior and adjust strategies accordingly."}
]

# Function to convert JSON file to DataFrame
def json_to_dataframe(json_input):
    try:
        with open(json_input, 'r') as json_file:
            data_list = json.load(json_file)
        df = pd.DataFrame(data_list)
        return df
    except Exception as e:
        #print(f"Error reading {json_input}: {e}")
        return pd.DataFrame()

def parse_with_fixer(response_content, parser):
    try:
        parsed_output = parser.parse(response_content)
        return True, parsed_output
    except Exception as e:
        #print(f"Initial parsing failed: {e}. Attempting to fix the output.")
        try:
            fixed_output = json.loads(response_content)  # Directly attempt to fix JSON format if it's a simple structure
            parsed_output = parser.parse_obj(fixed_output)  # Use parse_obj for Pydantic parsing
            return True, parsed_output
        except Exception as fix_e:
            #print(f"Fixing failed: {fix_e}")
            return False, None

# Function to add new data
def add_to_data_list(data, question, answer, head, text):
    data.append({
        'question': question,
        'answer': answer,
        'head': head,
        'text': text,
    })

# Function to parse JSON files in a directory and return a DataFrame
def parse_directory(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            df = json_to_dataframe(filepath)
            if not df.empty:
                data.append(df)
    if data:
        return pd.concat(data, ignore_index=True)
    else:
        raise ValueError("No JSON files found or parsed successfully in the directory.")

# Updated generate_questions function to accept number of questions as parameter and default to 3
def generate_questions(client, context, num_questions=3):
    prompt = PromptTemplate(
        template="Given the following context, generate {num_questions} questions that a nurse or caregiver might ask while taking care of a person with dementia. The questions you generate are scenario based as if you're looking for immediate help while caring for a person, and use informal and simple wording in the form of a casual conversation. \nContext: {context}\n{format_instructions}\nResponses should be in JSON format with no other text or attributes.",
        input_variables=["context", "num_questions"],
        partial_variables={"format_instructions": question_instructions}
    )
    completion = client.chat.completions.create(
        model=model,
        messages=few_shot_examples_questions + [{"role": "user", "content": prompt.format(context=context, num_questions=num_questions, format_instructions=question_instructions)}],
        temperature=1.7,
    )
    question_response = completion.choices[0].message.content
    valid, parsed_output = parse_with_fixer(question_response, json_parser)
    if valid:
        return [q['question'] for q in parsed_output]  # Extract the questions from the parsed output
    else:
        raise Exception("Invalid JSON output in generate_questions")

# Function to generate answers from the generated questions
def generate_answer(client, question):
    prompt = PromptTemplate(
        template="{question}",
        input_variables=["question"]
    )
    completion = client.chat.completions.create(
        messages=few_shot_examples_answers + [{"role": "user", "content": prompt.format(question=question)}],
        temperature=0.7,
        model=model,
    )
    #print(completion)
    answer_response = completion.choices[0].message.content
    #print(f"Answer Response: {answer_response}")
    return answer_response  # Return the raw answer response

# Function to calculate Overlap using cosine similarity
def calculate_overlap(question, context):
    vectorizer = CountVectorizer().fit_transform([question, context])
    vectors = vectorizer.toarray()
    return 1 - cosine(vectors[0], vectors[1])

# Function to calculate Diversity using edit distance
def calculate_diversity(questions):
    distances = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            distances.append(nltk.edit_distance(questions[i], questions[j]))
    return sum(distances) / len(distances) if distances else 0

# Function to count the number of tokens in questions and answers
def count_tokens(text):
    return len(word_tokenize(text))

# Function to process data and generate questions and answers with retries and custom number of questions
def process_data(df, client, num_retries=3, num_questions= 3, add_annotations=False):
    data = []

    def process_single_context(context, answer_retry_count=num_retries, question_retry_count=num_retries):
        try:
            # Generate questions with specified number
            question_data = generate_questions(client, context.get('Chunk', ''), num_questions)
            #print(f"Generated Questions: {question_data}")
            if not question_data:
                raise Exception("No questions generated")

            for question in question_data:
                answer_generated = False
                retries_left = answer_retry_count
                while not answer_generated and retries_left > 0:
                    try:
                        answer_data = generate_answer(RAG, question)
                        #print(f"Generated Answer: {answer_data}")
                        if not answer_data:
                            raise Exception("No answer generated")
                        answer_generated = True
                    except Exception as e:
                        retries_left -= 1
                        #print(f"Error generating answer. Retrying... ({retries_left} retries left)")
                        if retries_left == 0:
                            raise Exception(f"Failed to generate answer after retries: {e}")

                # overlap = calculate_overlap(question, context.get('Chunk', ''))
                # diversity = calculate_diversity(question_data)
                # details = count_tokens(question) + count_tokens(answer_data)

                add_to_data_list(data, question=question, answer=answer_data, head=context.get('Title', ''), text=context.get('Chunk', ''))

            return True
        except Exception as e:
            if question_retry_count > 0:
                #print(f"Error processing context. Retrying... ({question_retry_count} retries left)")
                return process_single_context(context, question_retry_count - 1)
            else:
                #print(f"Failed to process context after retries: {e}")
                return False

    # for i in range(len(df)):
    #     context = df.iloc[i]
    #     print(f"Processing context {i+1}/{len(df)}")
    #     print(f"Context: {context}")
    #     process_single_context(context, num_retries)
        # Use tqdm to display a progress bar for context processing
    for i in tqdm(range(len(df)), desc="Processing contexts"):
        context = df.iloc[i]
        process_single_context(context, num_retries)

    result_df = pd.DataFrame(data)
    # Add empty columns for annotations if needed
    if add_annotations:
        annotation_columns = ['Relevance', 'Global Relevance', 'Coverage', 'Succinctness', 'Fluency', 'Groundedness']
        for col in annotation_columns:
            result_df[col] = ''

    return result_df

# Main function to process the directory and generate questions and answers for all data
def process_all(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
        
    df = parse_directory(directory)
    result_df = process_data(df, client, add_annotations=False)
    return result_df

    directory_path = 'JSON' 
# Process all available data
result_all_df = process_all(directory_path)
result_all_df.to_csv('QA_3.1_8B_Q8.csv', index=False)