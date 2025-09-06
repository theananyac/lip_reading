from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_detection():
    # Run the lip reading detection and recording logic (reuse existing code)
    subprocess.run(["python", "record_video_on_trigger.py"])
    return render_template('index.html', result="Recording Triggered & Processed")

if __name__ == '__main__':
    app.run(debug=True)
