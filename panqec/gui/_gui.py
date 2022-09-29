import numpy as np

from flask import Flask, send_from_directory, request, json, render_template
from panqec.codes import (
    Toric2DCode, RotatedPlanar2DCode, Planar2DCode,
    Toric3DCode, RotatedPlanar3DCode, RhombicCode,
    Planar3DCode, RotatedToric3DCode, XCubeCode, Quasi2DCode,
    Color3DCode, Color666Code, Color488Code
)
from panqec.decoders import (
    MatchingDecoder, RotatedSweepMatchDecoder,
    ZMatchingDecoder, SweepMatchDecoder,
    BeliefPropagationOSDDecoder, MemoryBeliefPropagationDecoder,
    XCubeMatchingDecoder
)
from panqec.error_models import PauliErrorModel
from panqec.error_models import (
    DeformedXZZXErrorModel, DeformedXYErrorModel, DeformedRhombicErrorModel,
    DeformedColorErrorModel
)

import webbrowser
import argparse

codes = {'Toric 2D': Toric2DCode,
         'Planar 2D': Planar2DCode,
         'Rotated Planar 2D': RotatedPlanar2DCode,
         '6.6.6 Color Code': Color666Code,
         '4.8.8 Color Code': Color488Code,
         'Toric 3D': Toric3DCode,
         'Rotated Toric 3D': RotatedToric3DCode,
         'Rotated Planar 3D': RotatedPlanar3DCode,
         'Rhombic': RhombicCode,
         'Planar 3D': Planar3DCode,
         'XCube': XCubeCode,
         'Quasi 2D': Quasi2DCode,
         '3D Color Code': Color3DCode}

error_models = {'None': PauliErrorModel,
                'XZZX': DeformedXZZXErrorModel,
                'XY': DeformedXYErrorModel,
                'Rhombic': DeformedRhombicErrorModel,
                'Color': DeformedColorErrorModel}

noise_directions = {'Pure X': (1, 0, 0),
                    'Pure Y': (0, 1, 0),
                    'Pure Z': (0, 0, 1),
                    'Depolarizing': (1/3, 1/3, 1/3)}

decoders = {'BP-OSD': BeliefPropagationOSDDecoder,
            'MBP': MemoryBeliefPropagationDecoder,
            'SweepMatch': SweepMatchDecoder,
            'RotatedSweepMatch': RotatedSweepMatchDecoder,
            'Matching': MatchingDecoder,
            'Optimal ∞ bias': ZMatchingDecoder,
            'XCube Matching': XCubeMatchingDecoder}


class GUI():
    def __init__(self):
        self.app = Flask(__name__)

        self.codes = codes
        self.decoders = decoders
        self._code_names = {}

    @property
    def code_names(self):
        self._code_names = {
            '2d': [code[0] for code in self.codes.items()
                   if code[1].dimension == 2],
            '3d': [code[0] for code in self.codes.items()
                   if code[1].dimension == 3]
        }

        return self._code_names

    @property
    def decoder_names(self):
        self._decoder_names = list(self.decoders.keys())

        return self._decoder_names

    def add_code(self, code_class, code_name):
        self.codes[code_name] = code_class

    def add_decoder(self, decoder_class, decoder_name):
        self.decoders[decoder_name] = decoder_class

    def _instantiate_code(self, data):
        Lx = data['Lx']
        Ly = data['Ly']
        if 'Lz' in data:
            Lz = data['Lz']
        code_name = data['code_name']
        deformed_axis = data['deformed_axis']

        if code_name in self.code_names['2d']:
            code = codes[code_name](Lx, Ly, deformed_axis=deformed_axis)
        elif code_name in self.code_names['3d']:
            code = codes[code_name](Lx, Ly, Lz, deformed_axis=deformed_axis)
        else:
            raise ValueError(f'Code {code_name} not recognized')

        return code

    def run(self, port=5000):
        @self.app.route('/')
        def send_index():
            return render_template('index.html')

        @self.app.route('/2d')
        def send_index_2d():
            return render_template('gui.html')

        @self.app.route('/3d')
        def send_index_3d():
            return render_template('gui.html')

        @self.app.route('/main.css')
        def css():
            return send_from_directory('static/css', 'main.css')

        @self.app.route('/favicon.ico')
        def favicon():
            return send_from_directory('static', 'favicon.ico')

        @self.app.route('/js/<path:path>')
        def send_js(path):
            return send_from_directory('js', path)

        @self.app.route('/code-names', methods=['POST'])
        def send_code_names():
            dim = request.json['dimension']
            code_names = self.code_names[f'{dim}d']

            return json.dumps(code_names)

        @self.app.route('/decoder-names', methods=['POST'])
        def send_decoder_names():
            code_name = request.json['code_name']
            code_id = codes[code_name].__name__

            decoder_names = []

            for decoder_name, decoder_class in decoders.items():
                if (decoder_class.allowed_codes is None
                   or code_id in decoder_class.allowed_codes):
                    decoder_names.append(decoder_name)

            return json.dumps(decoder_names)

        @self.app.route('/code-data', methods=['POST'])
        def send_code_data():
            rotated_picture = request.json['rotated_picture']

            code = self._instantiate_code(request.json)

            qubits = [code.qubit_representation(location, rotated_picture)
                      for location in code.qubit_coordinates]
            stabilizers = [code.stabilizer_representation(location,
                                                          rotated_picture)
                           for location in code.stabilizer_coordinates]

            logical_z = code.logicals_z
            logical_x = code.logicals_x

            # print(code.stabilizer_matrix.toarray().tolist())

            return json.dumps({'H': code.stabilizer_matrix.toarray().tolist(),
                               'qubits': qubits,
                               'stabilizers': stabilizers,
                               'logical_z': logical_z.tolist(),
                               'logical_x': logical_x.tolist()})

        @self.app.route('/decode', methods=['POST'])
        def send_correction():
            content = request.json
            syndrome = np.array(content['syndrome'])
            p = content['p']
            noise_deformation = content['noise_deformation']
            max_bp_iter = content['max_bp_iter']
            alpha = content['alpha']
            beta = content['beta']
            decoder_name = content['decoder']
            error_model_name = content['error_model']

            code = self._instantiate_code(content)

            rx, ry, rz = noise_directions[error_model_name]
            error_model = error_models[noise_deformation](rx, ry, rz)

            kwargs = {}
            if decoder_name in ['BP-OSD', 'MBP']:
                kwargs['max_bp_iter'] = max_bp_iter
            if decoder_name == 'BP-OSD':
                kwargs['osd_order'] = 0
            if decoder_name == 'MBP':
                kwargs['alpha'] = alpha
                kwargs['beta'] = beta

            decoder = self.decoders[decoder_name](code, error_model, p,
                                                  **kwargs)

            correction = decoder.decode(syndrome)

            correction_x = correction[:code.n]
            correction_z = correction[code.n:]

            return json.dumps({'x': correction_x.tolist(),
                               'z': correction_z.tolist()})

        @self.app.route('/new-errors', methods=['POST'])
        def send_random_errors():
            content = request.json
            p = content['p']
            noise_deformation = content['noise_deformation']
            error_model_name = content['error_model']

            code = self._instantiate_code(content)

            rx, ry, rz = noise_directions[error_model_name]
            error_model = error_models[noise_deformation](rx, ry, rz)

            errors = error_model.generate(code, p)

            bsf_to_str_map = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z',
                              (1, 1): 'Y'}
            error_spec = [
                (
                    bsf_to_str_map[
                        (errors[i_qubit], errors[i_qubit + code.n])
                    ],
                    [
                        coords for index, coords in enumerate(
                            code.qubit_coordinates
                        )
                        if index == i_qubit
                    ][0]
                )
                for i_qubit in range(code.n)
            ]
            error_spec = [spec for spec in error_spec if spec[0] != 'I']
            return json.dumps(errors.tolist())

        self.app.run(port=port)


def open_browser(port):
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}/')


def run_gui():
    parser = argparse.ArgumentParser(description='Run GUI server')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port where to run the server')
    args = parser.parse_args()
    port = args.port

    gui = GUI()
    gui.run(port=port)


if __name__ == '__main__':
    run_gui()
