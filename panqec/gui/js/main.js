import { Interface } from './gui.js'

var defaultCode = codeDimension == 2 ? 'Planar 2D' : 'Toric 3D';
var defaultSize = codeDimension == 2 ? 4 : 4;

const params = {
    dimension: codeDimension,
    errorProbability: 0.1,
    Lx: defaultSize,
    Ly: defaultSize,
    Lz: defaultSize,
    noiseDeformationName: 'None',
    decoder: 'BP-OSD',
    max_bp_iter: 10,
    alpha: 0.4,
    beta: 0,
    channel_update: false,
    errorModel: 'Depolarizing',
    codeName: defaultCode,
    rotated: true,
    coprime: false,
    codeDeformationName: 'None'
};

const colors = {
    background: 0x102542
};

const keycode = {'d': 68, 'r': 82, 'backspace': 8, 'o': 79, 'x': 88, 'z': 90};

var gui = new Interface(params, colors, keycode);

gui.init()
gui.buildInstructions();
gui.buildReturnArrow();
gui.buildMenu();
gui.animate();
