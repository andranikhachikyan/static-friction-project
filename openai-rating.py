# With ChatGPT , it gets the objects from verified_words, rating them for specific variables listed in this algorithm, puts them in data_a.json file which is one of the required datas to run the algorithm
## Defined_words are not used, to improve this algorithm we can use defined words instead of verified words.
import json
import openai
import re

openai.api_key=''
# import open ai


# Replace 'your-api-key' with your actual OpenAI API key

# Define a function to get a response from GPT-3
def get_gpt3_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an advanced AI expert system specialized in evaluating and rating specific physical characteristics of objects relevant to static friction coefficients. Your primary function is to provide a single numerical rating out of 1000 for a specified physical property of an object. When presented with an object name and a specific physical characteristic, you should:  1. Provide a single rating out of 1000 for the specified physical characteristic from this list:    - Surface roughness    - Material hardness    - Surface cleanliness    - Material composition    - Elasticity/Young's modulus    - Crystalline structure    - Surface energy    - Surface oxide layer    - Moisture content    - Temperature    - Atomic-scale adhesion    - Surface coatings or treatments  2. Base your rating on the most common or typical physical appearance of the object if variations exist.  3. Provide only the numerical rating without any explanation or additional context.  4. If the specified characteristic is not applicable to the given object, provide a rating of -1.  5. Use your expertise to estimate the rating even for rare or unusual objects.  6. Respond with only the numerical rating, ensuring it is an integer between 0 and 1000.  Your response should consist of a single integer number between 0 and 1000, with no additional text or explanation."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()

# Load the original JSON file

with open('verified_words.json', 'r') as file:
    objects = json.load(file)

# Function to get user ratings
def get_rating(object_name, category):
    while True:
        try:
            rawrating = (get_gpt3_response(f"Give me an approximate rating for the object {object_name}'s {category} from (0-1000): "))
            rating = simplifying(rawrating)
            print(rawrating)

            if isinstance(rating, int):
                return rating
            else:
                return -1
        except ValueError:
            print("Please enter a valid number.")

# New dictionary to store rated objects
rated_objects = {}

def simplifying(rawrating):
    # Extract numeric characters from the string
    numeric_part = re.search(r'\d+', rawrating)
    
    if numeric_part:
        # Convert the extracted numeric part to an integer
        return int(numeric_part.group())
    else:
        # Return None or raise an error if no number is found
        return None  # or raise ValueError("No number found in the string")





# New dictionary to store rated objects
rated_objects = {}

# Rate each object

for obj in objects:
    print(f"\nRating: {obj}")
    rated_objects[obj] = {
        "Surface Roughness": get_rating(obj, "Surface Roughness"),
        "Material Hardness": get_rating(obj, "Material Hardness"),
        "Surface Cleanliness": get_rating(obj, "Surface Cleanliness"),
        "Material Composition": get_rating(obj, "Material Composition"),
        "Elasticity/Young's Modulus": get_rating(obj, "Elasticity/Young's Modulus"),
        "Crystalline Structure": get_rating(obj, "Crystalline Structure"),
        "Surface Energy": get_rating(obj, "Surface Energy"),
        "Surface Oxide Layer": get_rating(obj, "Surface Oxide Layer"),
        "Moisture Content": get_rating(obj, "Moisture Content"),
        "Temperature": get_rating(obj, "Temperature"),
        "Atomic-scale Adhesion": get_rating(obj, "Atomic-scale Adhesion"),
        "Surface Coatings or Treatments": get_rating(obj, "Surface Coatings or Treatments")
    }


# Save the new JSON file
with open('data_a.json', 'w') as file:
    json.dump(rated_objects, file, indent=4)

print("\nRating complete. New JSON file 'rated_objects.json' has been created.")

######################################





