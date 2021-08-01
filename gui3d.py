import numpy as np

from flask import Flask, send_from_directory, request, json
from bn3d.bp_os_decoder import bp_osd_decoder
from bn3d.tc3d import ToricCode3D, SweepMatchDecoder
from bn3d.bp_os_decoder import BeliefPropagationOSDDecoder
from bn3d.noise import PauliErrorModel
from bn3d.deform import DeformedPauliErrorModel, DeformedSweepMatchDecoder

import webbrowser
from threading import Timer

app = Flask(__name__)


def open_browser(port):
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}/')


@app.route('/')
def send_index():
    return send_from_directory('gui', 'index.html')


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('gui/js', path)


@app.route('/stabilizer-matrix', methods=['POST'])
def send_stabilizer_matrix():
    L = request.json['L']

    code = ToricCode3D(L, L, L)
    n_vertices = int(np.product(code.size))
    n_stabilizers = code.stabilizers.shape[0]
    n_faces = n_stabilizers - n_vertices
    n_qubits = code.n_k_d[0]

    Hz = code.stabilizers[:n_faces, :n_qubits]
    Hx = code.stabilizers[n_faces:, n_qubits:]

    return json.dumps({'Hx': Hx.tolist(), 'Hz': Hz.tolist()})


@app.route('/decode', methods=['POST'])
def send_correction():
    content = request.json
    syndrome = np.array(content['syndrome'])
    L = content['L']
    p = content['p']
    deformed = content['deformed']
    max_bp_iter = content['max_bp_iter']
    decoder_name = content['decoder']

    code = ToricCode3D(L, L, L)
    n_vertices = int(np.product(code.size))
    n_stabilizers = code.stabilizers.shape[0]
    n_faces = n_stabilizers - n_vertices
    n_qubits = code.n_k_d[0]

    if deformed:
        error_model = DeformedPauliErrorModel(1, 0, 0)
    else:
        error_model = PauliErrorModel(1, 0, 0)

    if decoder_name == 'bp':
        decoder = BeliefPropagationOSDDecoder(error_model, p,
                                              max_bp_iter=max_bp_iter,
                                              deformed=deformed)
    else:
        if deformed:
            decoder = DeformedSweepMatchDecoder(error_model, p)
        else:
            decoder = SweepMatchDecoder()

    correction = decoder.decode(code, syndrome)
    
    correction_x = correction[:n_qubits]
    correction_z = correction[n_qubits:]

    return json.dumps({'x': correction_x.tolist(), 'z': correction_z.tolist()})


@app.route('/new-errors', methods=['POST'])
def send_random_errors():
    content = request.json
    L = content['L']
    p = content['p']
    deformed = content['deformed']

    code = ToricCode3D(L, L, L)
    
    if deformed:
        error_model = DeformedPauliErrorModel(1, 0, 0)
    else:
        error_model = PauliErrorModel(1, 0, 0)
    errors = error_model.generate(code, p)

    return json.dumps(errors.tolist())


if __name__ == '__main__':
    port = 5000
    Timer(1, open_browser, [port]).start()

    app.run(port=port)
