import os
import sys
import io
import logging
import base64
import tempfile

from flask import Flask, render_template, request, jsonify
import numpy as np

# Silence TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)

app = Flask(__name__)

# Load model once at startup
hand = None

def load_model():
    global hand
    try:
        from demo import Hand
        hand = Hand()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model load error: {e}")
        hand = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    global hand
    if hand is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    text      = data.get('text', 'we made it').strip()
    style     = int(data.get('style', 0))
    bias      = float(data.get('bias', 0.75))       # legibility: 0=messy, 1=neat
    width     = float(data.get('width', 1.5))        # stroke width
    color     = data.get('color', '#1a1917')

    # Validate text
    import drawing as drw
    valid = set(drw.alphabet)
    cleaned = ''.join(c for c in text if c in valid)
    if not cleaned:
        return jsonify({'error': 'No valid characters in text'}), 400

    # Split into lines of max 75 chars at word boundaries
    words = cleaned.split(' ')
    lines = []
    current = ''
    for word in words:
        if not word:
            continue
        test = (current + ' ' + word).strip()
        if len(test) <= 75:
            current = test
        else:
            if current:
                lines.append(current)
            current = word[:75]
    if current:
        lines.append(current)

    if not lines:
        return jsonify({'error': 'Empty text'}), 400

    biases = [bias] * len(lines)
    styles = [style] * len(lines)
    colors = [color] * len(lines)
    widths = [width] * len(lines)

    # Generate to a temp SVG file
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmpname = tmp.name

    try:
        hand.write(
            filename=tmpname,
            lines=lines,
            biases=biases,
            styles=styles,
            stroke_colors=colors,
            stroke_widths=widths
        )
        with open(tmpname, 'r') as f:
            svg_content = f.read()
    finally:
        try:
            os.unlink(tmpname)
        except Exception:
            pass

    # Strip white background rect from SVG for transparent output
    svg_content = svg_content.replace(
        'fill="white"', 'fill="transparent"'
    )

    return jsonify({'svg': svg_content})


if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000)
