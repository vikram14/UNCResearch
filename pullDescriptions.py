from vectorize import load_json, save_json
import openai
from intellidiff import IntelliDiff
def getDesc(methods,name):
    differ=IntelliDiff()
    openai.api_key = "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"
    for i in range(0, len(methods), 20):
        print(f"{i}/{len(methods)}")
        try:
            code = [method['code']+"\n/* Explain what the previous function is doing: It" for method in methods[i:i+20]]
            response = openai.Completion.create(
            engine="code-davinci-001",
            prompt=code,
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0.6,
            presence_penalty=0
            )
            for j,method in  enumerate(methods[i:i+20]):
                method['description']="it" +' '.join(differ.getSentences(response['choices'][j].text))
            save_json(methods, r'C:\Users\vikram14\Desktop\Research\IntelliDiff'+f'\\{name}')
        except Exception as e:
            print(e)
            print(f"error at {i}")
            continue
