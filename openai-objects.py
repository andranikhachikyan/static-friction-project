#  With ChatGPT pulling the words from words_alpha.txt , asking if the word is an object or not, if yes it saves it to verified_words.json and defines the verified word and saves the word and its definition to defined_words.json
## Next algorithm is only using the verified_words.json file
### PROBLEM - It only finds the objects, but asphalt is not an object and still used in friction problems. 

import json
import openai

openai.api_key=''
# import open ai


# Replace 'your-api-key' with your actual OpenAI API key

# Define a function to get a response from GPT-3
def get_gpt3_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant specialized in providing detailed, accurate descriptions of physical objects. Your primary function is to act as a comprehensive dictionary that defines objects and their physical appearances. When presented with an object name or category, you should:  Provide a clear, concise definition of the object. Describe its typical physical characteristics, including:  Size and dimensions (average or range) Shape Color(s) Material composition Texture Any distinctive features or variations   If relevant, briefly mention:  Common uses or functions Notable variations or subcategories Any significant cultural or historical context   Avoid subjective judgments or personal opinions. If an object has multiple meanings or uses, briefly list them and focus on the most common interpretation unless otherwise specified. Use precise, descriptive language suitable for both general users and specialists. If asked about a very specific or rare object, indicate that the information provided is to the best of your knowledge but may not be exhaustive.  Respond concisely unless asked for more detail. Always prioritize accuracy and clarity in your descriptions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()

#to start off: function checker if a word is obj or not
#work in progress; not sure how to code openai stuff yet
def object_checker(word):
    prompt_yesno = f"is any definition of '{word}' an object? Respond either yes or no without any other signs like dot or comma "
    return get_gpt3_response(prompt_yesno) 

#getting definition
def get_definition(word):
    prompt_definitions = f"Define the object '{word}'"
    output=get_gpt3_response(prompt_definitions)

    return output



verified_objects = []
defined_words = {}


with open('/Users/dogac/Desktop/friction111/words_alpha10.txt', 'r') as file:
   words = file.readlines()


#checker word
for word in words:
   word = word.strip()
   print (object_checker(word))
   if "Yes" in object_checker(word):
       verified_objects.append(word)
       print(f'this {word} is an object')

for word in verified_objects:
   definition = get_definition(word)
   defined_words[word] = definition
   print (definition)

#save lists into json files
with open('verified_words.json', 'w') as json_file:
   json.dump(verified_objects, json_file)
   
with open('defined_words.json', 'w') as json_file:
   json.dump(defined_words, json_file)