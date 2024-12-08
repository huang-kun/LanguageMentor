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

def discover_course_goal(requirement_text, output_filename):
    '''开发英文课程目标和任务的提示'''
    model = 'gpt-4o-mini'

    project_dir = os.getcwd()
    find_goal_prompt_path = os.path.join(project_dir, 'prompts', 'discover_goal_prompt_en.txt')
    
    with open(find_goal_prompt_path, 'r') as f:
        system_prompt = f.read().strip()

    start_llm_conversation(
        model=model,
        system_prompt=system_prompt,
        start_chat=requirement_text,
        output=os.path.join(project_dir, 'temp_works', output_filename)
    )

def discover_rent_course_goal_homework():
    # 我想设计一个租房场景的英语课程，请让AI扮演一位房屋中介来指导用户学习英语。
    requirement_text = "I want to design an English course for a house rental scenario. " \
    "Please let AI play the role of a real estate agent to guide users to learn English."
    discover_course_goal(
        requirement_text=requirement_text,
        output_filename='rent_course_goal.txt'
    )

def discover_dating_course_goal_homework():
    requirement_text = "I want to design an English course for dating scenario. " \
    "Please let AI play the role of the dating coach to guide user to learn English."
    discover_course_goal(
        requirement_text=requirement_text,
        output_filename='dating_course_goal.txt'
    )

if __name__ == '__main__':
    discover_dating_course_goal_homework()