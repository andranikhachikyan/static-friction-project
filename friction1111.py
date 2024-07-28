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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()

#to start off: function checker if a word is obj or not
#work in progress; not sure how to code openai stuff yet
def object_checker(word):
    output_boolean_value = 1
    prompt_yesno = f"is any definition of '{word}' an object? Respond either yes or no without any other signs like dot or comma "

    return get_gpt3_response(prompt_yesno) 
# note: not

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
