import json
# import open ai


#to start off: function checker if a word is obj or not
#work in progress; not sure how to code openai stuff yet
def object_checker(word):
    output_boolean_value = 1
    prompt_yesno = "is the '{word}' an object? Respond either yes or no"

    return output_boolean_value 
# note: not


#getting definition
def get_definition(word):
    output_definition = 'This is an object to erase ink'
    prompt_definitions = "Define the word '{word}'"
    return output_definition


verified_objects = []
defined_words = {}


with open('/Users/dogac/Desktop/words_alpha.txt', 'r') as file:
   words = file.readlines()


#checker word
for word in words:
   word = word.strip()
   if object_checker(word) == 1 :
       verified_objects.append(word)

for word in verified_objects:
   definition = get_definition(word)
   defined_words[word] = definition




#save lists into json files
with open('verified_words.json', 'w') as json_file:
   json.dump(verified_objects, json_file)
with open('defined_words.json', 'w') as json_file:
   json.dump(defined_words, json_file)
