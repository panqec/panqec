import numpy as np

from flask import (
    Flask, send_from_directory, request, json, render_template
)
from panqec.codes import (
    Toric2DCode, RotatedPlanar2DCode, Planar2DCode,
    Toric3DCode, RotatedPlanar3DCode, RhombicToricCode, RhombicPlanarCode,
    Planar3DCode, RotatedToric3DCode, XCubeCode,
    HollowPlanar3DCode, HollowRhombicCode,
    Color3DCode, Color666PlanarCode, Color666ToricCode, Color488Code
)
from panqec.decoders import (
    MatchingDecoder, RotatedSweepMatchDecoder,
    SweepMatchDecoder, BeliefPropagationOSDDecoder,
    MemoryBeliefPropagationDecoder, XCubeMatchingDecoder
)
from panqec.error_models import PauliErrorModel

import webbrowser
import argparse

codes = {'Toric 2D': Toric2DCode,
         'Planar 2D': Planar2DCode,
         'Rotated Planar 2D': RotatedPlanar2DCode,
         '6.6.6 Color Code (toric)': Color666ToricCode,
         '6.6.6 Color Code (planar)': Color666PlanarCode,
         '4.8.8 Color Code': Color488Code,
         'Toric 3D': Toric3DCode,
         'Planar 3D': Planar3DCode,
         'Rotated Toric 3D': RotatedToric3DCode,
         'Rotated Planar 3D': RotatedPlanar3DCode,
         'Rhombic Toric 3D': RhombicToricCode,
         'Rhombic Planar 3D': RhombicPlanarCode,
         'XCube': XCubeCode,
         'Hollow Planar 3D': HollowPlanar3DCode,
         'Hollow Rhombic Code': HollowRhombicCode,
         '3D Color Code': Color3DCode}

noise_directions = {'Pure X': (1, 0, 0),
                    'Pure Y': (0, 1, 0),
                    'Pure Z': (0, 0, 1),
                    'Depolarizing': (1/3, 1/3, 1/3)}

decoders = {'BP-OSD': BeliefPropagationOSDDecoder,
            'MBP': MemoryBeliefPropagationDecoder,
            'SweepMatch': SweepMatchDecoder,
            'RotatedSweepMatch': RotatedSweepMatchDecoder,
            'Matching': MatchingDecoder,
            'XCube Matching': XCubeMatchingDecoder}


class GUI():
    app = None

    def __init__(self):
        self.app = Flask(__name__)

        self.codes = codes
        self.decoders = decoders
        self._code_names = {}

        self.add_all_routes()

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
        deformation_name = data['code_deformation_name']

        if code_name in self.code_names['2d']:
            code = codes[code_name](Lx, Ly)
        elif code_name in self.code_names['3d']:
            code = codes[code_name](Lx, Ly, Lz)
        else:
            raise ValueError(f'Code {code_name} not recognized')

        if deformation_name != "None":
            code.deform(deformation_name)

        return code

    def run(self, *args, **kwargs):
        self.app.run(*args, **kwargs)

    def add_all_routes(self):
        self.app.add_url_rule(
            "/", "send_index",
            self.send_index
        )
        self.app.add_url_rule(
            "/main.css", "css",
            self.css
        )
        self.app.add_url_rule(
            "/favicon.ico", "favicon",
            self.favicon
        )
        self.app.add_url_rule(
            "/js/<path:path>", "send_js",
            self.send_js
        )
        self.app.add_url_rule(
            "/2d", "send_index_2d",
            self.send_index_2d
        )
        self.app.add_url_rule(
            "/3d", "send_index_3d",
            self.send_index_3d
        )
        self.app.add_url_rule(
            "/code-names", "code-names",
            self.send_code_names, methods=['POST']
        )
        self.app.add_url_rule(
            "/decoder-names", "decoder-names",
            self.send_decoder_names, methods=['POST']
        )
        self.app.add_url_rule(
            "/deformation-names", "deformation-names",
            self.send_deformation_names, methods=['POST']
        )
        self.app.add_url_rule(
            "/code-data", "code-data",
            self.send_code_data, methods=['POST']
        )
        self.app.add_url_rule(
            "/decode", "decode",
            self.send_correction, methods=['POST']
        )
        self.app.add_url_rule(
            "/new-errors", "new-errors",
            self.send_random_errors, methods=['POST']
        )

    def send_index_2d(self):
        return render_template('gui.html')

    def send_index_3d(self):
        return render_template('gui.html')

    def send_index(self):
        return render_template('index.html')

    def css(self):
        return send_from_directory('static/css', 'main.css')

    def favicon(self):
        return send_from_directory('static', 'favicon.ico')

    def send_js(self, path):
        return send_from_directory('js', path)

    def send_code_names(self):
        dim = request.json['dimension']
        code_names = self.code_names[f'{dim}d']

        return json.dumps(code_names)

    def send_decoder_names(self):
        code_name = request.json['code_name']
        code_id = codes[code_name].__name__

        decoder_names = []

        for decoder_name, decoder_class in decoders.items():
            if (decoder_class.allowed_codes is None
                    or code_id in decoder_class.allowed_codes):
                decoder_names.append(decoder_name)

        return json.dumps(decoder_names)

    def send_deformation_names(self):
        code_name = request.json['code_name']
        deformation_names = codes[code_name].deformation_names

        return json.dumps(deformation_names)

    def send_code_data(self):
        rotated_picture = request.json['rotated_picture']

        code = self._instantiate_code(request.json)

        qubits = [code.qubit_representation(location, rotated_picture)
                  for location in code.qubit_coordinates]
        stabilizers = [code.stabilizer_representation(location,
                                                      rotated_picture)
                       for location in code.stabilizer_coordinates]

        logical_z = code.logicals_z
        logical_x = code.logicals_x

        return json.dumps({'H': code.stabilizer_matrix.toarray().tolist(),
                           'qubits': qubits,
                           'stabilizers': stabilizers,
                           'logical_z': logical_z.tolist(),
                           'logical_x': logical_x.tolist()
                           })

    def send_correction(self):
        content = request.json
        syndrome = np.array(content['syndrome'])
        p = content['p']
        noise_deformation = content['noise_deformation_name']
        max_bp_iter = content['max_bp_iter']
        alpha = content['alpha']
        beta = content['beta']
        decoder_name = content['decoder']
        error_model_name = content['error_model']

        if noise_deformation == 'None':
            noise_deformation = None

        code = self._instantiate_code(content)

        rx, ry, rz = noise_directions[error_model_name]
        error_model = PauliErrorModel(rx, ry, rz, noise_deformation)

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

    def send_random_errors(self):
        content = request.json
        p = content['p']
        noise_deformation = content['noise_deformation_name']
        error_model_name = content['error_model']

        if noise_deformation == 'None':
            noise_deformation = None

        code = self._instantiate_code(content)

        rx, ry, rz = noise_directions[error_model_name]
        error_model = PauliErrorModel(rx, ry, rz, noise_deformation)

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


def open_browser(port):
    webbrowser.open_new_tab(f'http://127.0.0.1:{port}/')


def run_gui():
    parser = argparse.ArgumentParser(description='Run GUI server')
    parser.add_argument('--port', '-p', type=int, default=5000,
                        help='Port where to run the server')
    args = parser.parse_args()
    port = args.port

    gui = GUI()
    # gui.add_all_endpoints()
    gui.run(port=port)


if __name__ == '__main__':
    run_gui()
