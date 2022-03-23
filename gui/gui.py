import numpy as np

from flask import Flask, send_from_directory, request, json, render_template
from bn3d.models import (
    Toric2DCode, RotatedPlanar2DCode, Planar2DCode,
    Toric3DCode, RotatedPlanar3DCode, RhombicCode,
    Planar3DCode, RotatedToric3DCode, XCubeCode
)
from bn3d.decoders import (
    Toric2DPymatchingDecoder, RotatedSweepMatchDecoder,
    RotatedInfiniteZBiasDecoder, SweepMatchDecoder, Toric2DPymatchingDecoder
)
from bn3d.decoders import BeliefPropagationOSDDecoder, MemoryBeliefPropagationDecoder, DeformedSweepMatchDecoder
from bn3d.noise import PauliErrorModel
from bn3d.error_models import (
    DeformedXZZXErrorModel, DeformedXYErrorModel, DeformedRhombicErrorModel
)

import webbrowser
import argparse

code_names = {'2d': ['toric-2d', 'rotated-planar-2d', 'planar-2d'],
              '3d': ['toric-3d', 'planar-3d', 'rotated-planar-3d', 'rhombic', 'rotated-toric-3d', 'xcube']}

code_class = {'toric-2d': Toric2DCode, 'planar-2d': Planar2DCode, 'rotated-planar-2d': RotatedPlanar2DCode,
              'toric-3d': Toric3DCode, 'rotated-toric-3d': RotatedToric3DCode, 'rotated-planar-3d': RotatedPlanar3DCode,
              'rhombic': RhombicCode, 'planar-3d': Planar3DCode, 'xcube': XCubeCode}

error_model_class = {'None': PauliErrorModel,
                     'XZZX': DeformedXZZXErrorModel,
                     'XY': DeformedXYErrorModel,
                     'Rhombic': DeformedRhombicErrorModel}

noise_directions = {'Pure X': (1, 0, 0),
                    'Pure Y': (0, 1, 0),
                    'Pure Z': (0, 0, 1),
                    'Depolarizing': (1/3, 1/3, 1/3)}


app = Flask(__name__)


def open_browser(port):
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}/')


@app.route('/')
def send_index():
    return render_template('index.html')


@app.route('/2d')
def send_index_2d():
    return render_template('gui.html')


@app.route('/3d')
def send_index_3d():
    return render_template('gui.html')


@app.route('/main.css')
def css():
    return send_from_directory('static/css', 'main.css')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/stabilizer-matrix', methods=['POST'])
def send_stabilizer_matrix():
    Lx = request.json['Lx']
    Ly = request.json['Ly']
    if 'Lz' in request.json:
        Lz = request.json['Lz']
    code_name = request.json['code_name']
    deformed_axis = request.json['deformed_axis']

    deformed_axis = {'x': 0, 'y': 1, 'z': 2, 'None': None}[deformed_axis]

    if code_name in code_names['2d']:
        code = code_class[code_name](Lx, Ly, deformed_axis=deformed_axis)
    elif code_name in code_names['3d']:
        code = code_class[code_name](Lx, Ly, Lz, deformed_axis=deformed_axis)
    else:
        raise ValueError(f'Code {code_name} not recognized')

    n_qubits = code.n

    Hz = code.stabilizers[:, :n_qubits]
    Hx = code.stabilizers[:, n_qubits:]

    qubit_index = code.qubit_index
    qubit_index = {str(list(coord)): i for coord, i in qubit_index.items()}

    vertex_index = code.vertex_index

    face_index = code.face_index
    face_index = {str(list(coord)): i for coord, i in face_index.items()}
    n_faces = len(face_index.keys())
    vertex_index = {
        str(list(coord)): i + n_faces for coord, i in vertex_index.items()
    }

    stabilizer_index = {
        'vertex': vertex_index, 'face': face_index
    }

    n_qubits = code.n
    logical_z = code.logical_zs
    logical_x = code.logical_xs

    return json.dumps({'Hx': Hx.toarray().tolist(),
                       'Hz': Hz.toarray().tolist(),
                       'qubit_index': qubit_index,
                       'stabilizer_index': stabilizer_index,
                       'logical_z': logical_z.toarray().tolist(),
                       'logical_x': logical_x.toarray().tolist()})


@app.route('/decode', methods=['POST'])
def send_correction():
    content = request.json
    syndrome = np.array(content['syndrome'])
    Lx = content['Lx']
    Ly = content['Ly']
    if 'Lz' in content:
        Lz = content['Lz']
    p = content['p']
    noise_deformation = content['noise_deformation']
    max_bp_iter = content['max_bp_iter']
    alpha = content['alpha']
    decoder_name = content['decoder']
    error_model_name = content['error_model']
    code_name = content['code_name']
    deformed_axis = content['deformed_axis']

    deformed_axis = {'x': 0, 'y': 1, 'z': 2, 'None': None}[deformed_axis]

    if code_name in code_names['2d']:
        code = code_class[code_name](Lx, Ly, deformed_axis=deformed_axis)
    elif code_name in code_names['3d']:
        code = code_class[code_name](Lx, Ly, Lz, deformed_axis=deformed_axis)
    else:
        raise ValueError(f'Code {code_name} not recognized')

    n_qubits = code.n

    if error_model_name in noise_directions.keys():
        rx, ry, rz = noise_directions[error_model_name]
    else:
        raise ValueError(f'Error model {error_model_name} not recognized')

    if noise_deformation in error_model_class.keys():
        error_model = error_model_class[noise_deformation](rx, ry, rz)
    else:
        raise ValueError(f'Deformation {noise_deformation} not recognized')

    if decoder_name == 'bp-osd':
        decoder = BeliefPropagationOSDDecoder(error_model, p,
                                              max_bp_iter=max_bp_iter,
                                              joschka=False)
    elif decoder_name == 'bp-osd-2':
        decoder = BeliefPropagationOSDDecoder(error_model, p,
                                              max_bp_iter=max_bp_iter,
                                              joschka=True)
    elif decoder_name == 'mbp':
        decoder = MemoryBeliefPropagationDecoder(error_model, p,
                                                 max_bp_iter=max_bp_iter,
                                                 alpha=alpha)
    elif decoder_name == 'matching':
        decoder = Toric2DPymatchingDecoder()
    elif decoder_name == 'sweepmatch':
        if "Rotated" in code.label:
            decoder = RotatedSweepMatchDecoder()
        elif noise_deformation == "XZZX":
            decoder = DeformedSweepMatchDecoder(error_model, p)
        elif noise_deformation == "None":
            decoder = SweepMatchDecoder()
        elif noise_deformation == "XY":
            raise NotImplementedError("No SweepMatch decoder for XY code")
        else:
            raise ValueError("Deformation not recognized")
    elif decoder_name == 'infzopt':
        decoder = RotatedInfiniteZBiasDecoder()
    else:
        raise ValueError(f'Decoder {decoder_name} not recognized')

    correction = decoder.decode(code, syndrome)

    correction_x = correction[:n_qubits]
    correction_z = correction[n_qubits:]

    return json.dumps({'x': correction_x.tolist(), 'z': correction_z.tolist()})


@app.route('/new-errors', methods=['POST'])
def send_random_errors():
    content = request.json
    Lx = content['Lx']
    Ly = content['Ly']
    if 'Lz' in content:
        Lz = content['Lz']
    p = content['p']
    noise_deformation = content['noise_deformation']
    error_model_name = content['error_model']
    code_name = content['code_name']
    deformed_axis = content['deformed_axis']

    deformed_axis = {'x': 0, 'y': 1, 'z': 2, 'None': None}[deformed_axis]

    if code_name in code_names['2d']:
        code = code_class[code_name](Lx, Ly, deformed_axis=deformed_axis)
    elif code_name in code_names['3d']:
        code = code_class[code_name](Lx, Ly, Lz, deformed_axis=deformed_axis)
    else:
        raise ValueError(f'Code {code_name} not recognized')

    if error_model_name in noise_directions.keys():
        rx, ry, rz = noise_directions[error_model_name]
    else:
        raise ValueError(f'Error model {error_model_name} not recognized')

    if noise_deformation in error_model_class.keys():
        error_model = error_model_class[noise_deformation](rx, ry, rz)
    else:
        raise ValueError(f'Deformation {noise_deformation} not recognized')

    errors = error_model.generate(code, p)

    n_qubits = code.n
    bsf_to_str_map = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}
    error_spec = [
        (
            bsf_to_str_map[
                (errors[i_qubit], errors[i_qubit + n_qubits])
            ],
            [
                coords for coords, index in code.qubit_index.items()
                if index == i_qubit
            ][0]
        )
        for i_qubit in range(n_qubits)
    ]
    error_spec = [spec for spec in error_spec if spec[0] != 'I']
    return json.dumps(errors.tolist())


def run_gui():
    parser = argparse.ArgumentParser(description='Run GUI server')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port where to run the server')
    args = parser.parse_args()
    port = args.port

    app.run(port=port)


if __name__ == '__main__':
    run_gui()