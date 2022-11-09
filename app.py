from flask import Flask, redirect, render_template, request, url_for

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
    # return "Hi there! WOW"
    return render_template('index.html')


@app.route('/<int:mark>')
def mark(mark):
    if mark > 50:
        result = "gj"
    else:
        result = "bj"
    # return result + ": " + str(mark)
    # url_for(name_of_def,its_prams)
    return redirect(url_for(result, score=mark))


@ app.route('/submit', methods=['POST', 'GET'])
def submit():
    mark = 0
    if request.method == 'POST':
        mark = float(request.form['mark'])
    if mark > 50:
        return redirect(url_for("mark", mark=mark))
    return redirect(url_for("mark", mark=mark))


@ app.route('/gj/<int:score>')
def gj(score):
    return render_template('res.html', score=score, res="GJ")


@ app.route('/BJ/<int:score>')
def bj(score):
    return render_template('res.html', score=score, res="BJ")


if __name__ == '__main__':
    app.run(debug=True)
