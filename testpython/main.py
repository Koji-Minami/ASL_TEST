from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
    Part,
    SafetySetting
)

import os,base64,json
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv('REGION')
PROJECT=os.getenv('GOOGLE_CLOUD_PROJECT')




def generate(audiofile,pdffile):
    print("開始")
    vertexai.init(project="qwiklabs-asl-02-26483600cdad", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-pro-002",
    )

    # audio_byte = audiofile.read()
    audio = Part.from_data(
    mime_type='audio/mpeg',
    data= audiofile
    )

    # pdf_bytes = pdffile.read()
    pdf = Part.from_data(
            mime_type="application/pdf",
            data = pdffile
    )



    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "response_mime_type": "application/json"
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]
    responses = model.generate_content(
    [pdf,audio, text],
    generation_config=generation_config,
    safety_settings=safety_settings,
    )



    return json.loads(responses.text)

with open('records.mp3','rb') as f:
    audio_bytes = f.read() 

with open('GazoMitsu.pdf','rb') as f:
    pdf_bytes = f.read()

intro = open('./prompt/intro.txt', 'r', encoding='UTF-8').read()
task1 = open('./prompt/task1.txt', 'r', encoding='UTF-8').read()
task2 = open('./prompt/task2.txt', 'r', encoding='UTF-8').read()
task3 = open('./prompt/task3.txt', 'r', encoding='UTF-8').read()
output_format = open('./prompt/output_format.txt', 'r', encoding='UTF-8').read()


text = f'''{intro} {task1} {task2} {task3} {output_format}'''


# first_res = generate(audio_bytes,pdf_bytes)

ideal_result = open('./manager_prompt/ideal_result.txt', 'r', encoding='UTF-8').read()
first_res = open('./firstres.txt', 'r', encoding='UTF-8').read()

manager_prompt_intro = ''' 
あなたはプロンプトエンジニアです。下記は理想的なOutput、現状の生成されたOutput、使用したプロンプトです。
理想的なOutputが出力されるように、プロンプトを調整してください。
Chain of Thoughtなどを有効活用してください。
プロンプトはTaskごとに分かれています。修正の必要がない場合はNoneとしてください。
また各Taskにはどの程度理想のOutputに近くなければならないかの方針があります。
# 出力形式 
{"intro":修正したプロンプト,"task1":修正したプロンプト,...,"output_format":修正したプロンプト}
'''

manager_prompt_constitute = '''
下記方針に従ってプロンプトの修正をしてください
Critical: 内容は必ず一致。表現の揺れも不可。
Moderate: 内容は概ね一致。表現の揺れは許容。
Negligible: 内容が大きくずれていなければ許容。

task1:Moderate,
task2:Critical
task3:Negligible
'''

manager_prompt = f'''
{manager_prompt_intro}
{manager_prompt_constitute}

# 理想的なOutput
{ideal_result}

# 現状のOutput
{first_res}

# 使用したプロンプト
{text}
'''

manager_model = GenerativeModel(
    "gemini-1.5-pro-002",
)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "response_mime_type": "application/json"
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]


responses = manager_model.generate_content(
    [manager_prompt],
    generation_config=generation_config,
    safety_settings=safety_settings,
    stream=True
    )

for response in responses:
    print(response.text, end="",flush=True)
