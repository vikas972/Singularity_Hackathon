import os
import openai
from flask import Flask, redirect, render_template, request, url_for, Response, jsonify

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

openai.api_key = os.getenv("OPENAI_API_KEY")
from camera import *

emotion = "https://youtu.be/2Vv-BfVoq4g"
photo = "https://content.wepik.com/statics/8773130/preview-page0.jpg"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    global emotion
    global photo
    if request.method == "POST":
        text = request.form["question"]
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=generate_prompt(text),
            temperature=0.8,
            max_tokens=150
        )
        if response.choices:
            result = response.choices[0].text
        frame,emotion,photo = VideoCamera().get_frame()
    return render_template("index.html", result=result,emotion = emotion, photo=photo)



# @app.route("/", methods=("GET", "POST"))
# def index():
#     if request.method == "POST":
#         text = request.form["question"]
#         response = openai.Completion.create(
#             model="text-davinci-002",
#             prompt=generate_prompt(text),
#             temperature=0.6,
#             max_tokens=150
#         )
#         # response = openai.Image.create(
#         #     prompt=animal,
#         #     n=1,
#         #     size="1024x1024"
#         #     )
#         # image_url = response['data'][0]['url']
#         print(response.choices)
#         print(response)
#         # frame,emotion = camera.get_frame()
#         return redirect(url_for("index", result=response.choices[0].text))

#     result = request.args.get("result")
#     print(result)
#     return render_template("index.html", result=result)

def gen(camera):
    while True:
        frame,emotion,p = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def generate_prompt(text):
    return """You are Mental Wellbeing assistent, Please make your response according to that:
Input: What are some simple ways to improve my mood?
Output: Spend time outdoors, practice gratitude, do enjoyable activities, connect with loved ones, and get enough sleep.
Input: How can I improve my self-esteem?
Output: Practice self-care, set realistic goals, challenge negative self-talk, focus on strengths, and seek support.
Input: {}
Output:""".format(
        text.capitalize()
    )
    # print(text)
    # if type(text)==int:
    #     text = '''Tell me a funny Joke'''





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
