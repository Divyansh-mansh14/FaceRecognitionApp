from flask import Flask
from app import views

app = Flask(__name__) # webserver gateway interphase (WSGI)

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,
                 methods=['GET','POST'])

# app.add_url_rule(rule='/app/age/',
#                  endpoint='age',
#                  view_func=views.ageapp,
#                  methods=['GET','POST'])

app.add_url_rule(rule='/app/facerecognition/', 
                 endpoint='facerecognition', 
                 view_func=views.facerecognition,
                 methods=['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)