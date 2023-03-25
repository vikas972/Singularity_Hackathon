import os

import openai
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        text = request.form["question"]
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt(text),
            temperature=0.6,
            max_tokens=150
        )
        # response = openai.Image.create(
        #     prompt=animal,
        #     n=1,
        #     size="1024x1024"
        #     )
        # image_url = response['data'][0]['url']
        print(response.choices)
        print(response)
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    print(result)
    return render_template("index.html", result=result)



def generate_prompt(text):
    print(text)
    # if type(text)==int:
    #     text = '''Tell me a funny Joke'''
    return text




# def generate_prompt(animal):
#     return """Suggest two names for an animal that is a superhero.

# Animal: Cat
# Names: Captain Sharpclaw, Agent Fluffball
# Animal: Dog
# Names: Ruff the Protector, Wonder Canine
# Animal: {}
# Names:""".format(
#         animal.capitalize()
#     )
