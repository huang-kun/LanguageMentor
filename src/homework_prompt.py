from openai import OpenAI
import httpx
import os
import textwrap

def create_llm_client():
    api_key = os.environ["OPENAI_API_KEY"]
    if "OPENAI_BASE_URL" in os.environ:
        base_url = os.environ["OPENAI_BASE_URL"]
        return OpenAI(
            base_url=base_url, 
            api_key=api_key,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
            ),
        )
    else:
        return OpenAI(api_key=api_key)

def get_llm_message(chat_completion):
    return chat_completion.choices[0].message.content

def test_llm_completion(client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )
    return completion

def test_llm():
    client = create_llm_client()
    completion = test_llm_completion(client)
    print(completion)

def start_llm_conversation(model='gpt-3.5-turbo', system_prompt='You are a helpful assistant.', start_chat=None, output=None, debug=False):
    client = create_llm_client()
    messages = [{'role': 'system', 'content': system_prompt}]
    loop_times = 0

    while True:
        loop_times += 1
        if start_chat and loop_times == 1:
            user_msg = start_chat
        else:
            user_msg = input("You: ")
            if not user_msg:
                continue
            if user_msg in ['q', 'quit', 'exit']:
                break

        messages.append({'role': 'user', 'content': user_msg})
        chat_comp = client.chat.completions.create(model=model, messages=messages)
        if debug:
            print("-------")
            print(messages)
            print(chat_comp)
            print("-------")

        llm_msg = get_llm_message(chat_comp)
        messages.append({'role': 'assistant', 'content': llm_msg})
        print(f"AI: {llm_msg}")

    if output:
        messages.pop(0)
        conversations = list(map(lambda x: f"{x['role']}: {x['content']}\n", messages))
        with open(output, 'w') as f:
            f.writelines(conversations)

def do_homework():
    model = 'gpt-4o-mini'

    project_dir = os.getcwd()
    engineer_prompt_path = os.path.join(project_dir, 'prompts', 'prompt_engineer_prompt.txt')
    old_teacher_prompt_path = os.path.join(project_dir, 'prompts', 'old_conversation_prompt.txt')
    
    with open(engineer_prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    with open(old_teacher_prompt_path, 'r') as f:
        teacher_prompt = f.read().strip()

    start_chat = textwrap.dedent('''
        Please improve the prompt for AI English teacher with few updates:
        - Generate 3 English example sentences to facilitate the conversation.
        - Provide a Format that structure reply content of the AI, including teaching review, example sentences, and AI role-played charactor replies.
        - DO NOT make big changes for original prompt, just make some adjustments to make sure the generated result from this prompt will be stable.
        ---- Here is the original prompt inside of the dash line area ----\n''')
    
    start_chat += teacher_prompt
    start_chat += "\n-------------------"
    
    start_llm_conversation(
        model=model, 
        system_prompt=system_prompt,
        start_chat=start_chat,
        output=os.path.join(project_dir, 'conversation', 'homework_chat.txt')
    )

if __name__ == '__main__':
    do_homework()