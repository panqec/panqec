import numpy as np

from flask import Flask, send_from_directory, request, json, render_template
from bn3d.models import (
    ToricCode3D, RotatedPlanarCode3D, RotatedToricCode3D, RhombicCode, PlanarCode3D
)
from bn3d.decoders import (
    Toric2DPymatchingDecoder, RotatedSweepMatchDecoder,
    RotatedInfiniteZBiasDecoder, SweepMatchDecoder
)
from qecsim.models.toric import ToricCode
from bn3d.decoders import BeliefPropagationOSDDecoder, DeformedSweepMatchDecoder
from bn3d.noise import PauliErrorModel
from bn3d.error_models import (
    DeformedXZZXErrorModel, DeformedXYErrorModel, DeformedRhombicErrorModel
)

import webbrowser

code_names = {'2d': ['toric-2d'],
              '3d': ['toric-3d', 'planar-3d', 'rotated-toric-3d', 'rotated-planar-3d', 'rhombic']}

code_class = {'toric-2d': ToricCode, 'toric-3d': ToricCode3D,
              'rotated-planar-3d': RotatedPlanarCode3D, 'rotated-toric-3d': RotatedToricCode3D,
              'rhombic': RhombicCode, 'planar-3d': PlanarCode3D}

error_model_class = {'None': PauliErrorModel,
                     'XZZX': DeformedXZZXErrorModel,
                     'XY': DeformedXYErrorModel,
                     'Rhombic': DeformedRhombicErrorModel}

noise_directions = {'Pure X': (1, 0, 0),
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

    indices = {}

    if code_name in code_names['2d']:
        code = code_class[code_name](Lx, Ly)

        n_qubits = code.n_k_d[0]
        n_stabilizers = code.stabilizers.shape[0]
        n_vertices = int(np.product(code.size))
        n_faces = n_stabilizers - n_vertices

        Hz = code.stabilizers[:n_faces, n_qubits:]
        Hx = code.stabilizers[n_faces:, :n_qubits]

    elif code_name in code_names['3d']:
        code = code_class[code_name](Lx, Ly, Lz)

        Hz = code.Hz
        Hx = code.Hx

        qubit_index = code.qubit_index
        qubit_index = {str(list(coord)): i for coord, i in qubit_index.items()}

        vertex_index = code.vertex_index
        vertex_index = {
            str(list(coord)): i for coord, i in vertex_index.items()
        }

        face_index = code.face_index
        face_index = {str(list(coord)): i for coord, i in face_index.items()}

        indices = {
            'qubit': qubit_index, 'vertex': vertex_index, 'face': face_index
        }

    n_qubits = code.n_k_d[0]
    logical_z = code.logical_zs
    logical_x = code.logical_xs

    return json.dumps({'Hx': Hx.tolist(),
                       'Hz': Hz.tolist(),
                       'indices': indices,
                       'logical_z': logical_z[:, n_qubits:].tolist(),
                       'logical_x': logical_x[:, :n_qubits].tolist()})


@app.route('/decode', methods=['POST'])
def send_correction():
    content = request.json
    syndrome = np.array(content['syndrome'])
    Lx = content['Lx']
    Ly = content['Ly']
    if 'Lz' in content:
        Lz = content['Lz']
    p = content['p']
    deformation = content['deformation']
    max_bp_iter = content['max_bp_iter']
    decoder_name = content['decoder']
    error_model_name = content['error_model']
    code_name = content['code_name']

    if code_name in code_names['2d']:
        code = code_class[code_name](Lx, Ly)
    elif code_name in code_names['3d']:
        code = code_class[code_name](Lx, Ly, Lz)
    else:
        raise ValueError(f'Code {code_name} not recognized')

    n_qubits = code.n_k_d[0]

    if error_model_name in noise_directions.keys():
        rx, ry, rz = noise_directions[error_model_name]
    else:
        raise ValueError(f'Error model {error_model_name} not recognized')

    if deformation in error_model_class.keys():
        error_model = error_model_class[deformation](rx, ry, rz)
    else:
        raise ValueError(f'Deformation {deformation} not recognized')

    if decoder_name == 'bp-osd':
        decoder = BeliefPropagationOSDDecoder(error_model, p,
                                              max_bp_iter=max_bp_iter,
                                              joschka=False)
    elif decoder_name == 'bp-osd-2':
        decoder = BeliefPropagationOSDDecoder(error_model, p,
                                              max_bp_iter=max_bp_iter,
                                              joschka=True)
    elif decoder_name == 'matching':
        decoder = Toric2DPymatchingDecoder()
    elif decoder_name == 'sweepmatch':
        if "Rotated" in code.label:
            decoder = RotatedSweepMatchDecoder()
        elif deformation == "XZZX":
            decoder = DeformedSweepMatchDecoder(error_model, p)
        elif deformation == "None":
            decoder = SweepMatchDecoder()
        elif deformation == "XY":
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
    deformation = content['deformation']
    error_model_name = content['error_model']
    code_name = content['code_name']

    if code_name in code_names['2d']:
        code = code_class[code_name](Lx, Ly)
    elif code_name in code_names['3d']:
        code = code_class[code_name](Lx, Ly, Lz)
    else:
        raise ValueError(f'Code {code_name} not recognized')

    if error_model_name in noise_directions.keys():
        rx, ry, rz = noise_directions[error_model_name]
    else:
        raise ValueError(f'Error model {error_model_name} not recognized')

    if deformation in error_model_class.keys():
        error_model = error_model_class[deformation](rx, ry, rz)
    else:
        raise ValueError(f'Deformation {deformation} not recognized')

    errors = error_model.generate(code, p)

    n_qubits = code.n_k_d[0]
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


if __name__ == '__main__':
    port = 5000
    # Timer(1, open_browser, [port]).start()

    app.run(port=port)
