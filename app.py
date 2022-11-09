from flask import Flask, redirect, url_for

"""
this is the WSGL applicaiton,
which we interact with,
standard to communicate with the server
"""

app = Flask(__name__)


"""
this a decorator,
it takes us to this url
"""


@app.route('/')
def welcome():
    return "Hi there! WOW"


@app.route('/<int:mark>')
def mark(mark):
    if mark > 50:
        result = "gj"
    else:
        result = "bj"
    # return result + ": " + str(mark)
    # url_for(name_of_def,its_prams)
    return redirect(url_for(result, score=mark))


@ app.route('/gj/<int:score>')
def gj(score):
    return str(score)


@ app.route('/bj/<int:score>')
def bj(score):
    return (score)


if __name__ == '__main__':
    app.run(debug=True)
