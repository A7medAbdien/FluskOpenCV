from flask import Flask

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


if __name__ == '__main__':
    app.run(debug=True)
