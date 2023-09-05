import { Interface } from './gui.js'

var defaultCode = codeDimension == 2 ? 'Toric 2D' : 'Toric 3D';
var defaultSize = codeDimension == 2 ? 4 : 2;

const params = {
    dimension: codeDimension,
    errorProbability: 0.05,
    Lx: 2,
    Ly: 2,
    Lz: 2,
    noiseDeformationName: 'None',
    decoder: 'BP-OSD',
    max_bp_iter: 20,
    alpha: 0.4,
    beta: 0,
    channel_update: false,
    errorModel: 'Pure X',
    codeName: defaultCode,
    rotated: false,
    coprime: false,
    codeDeformationName: 'None'
};

const colors = {
    background: 0x102542
};

const keycode = {};

var gui = new Interface(params, colors, keycode);

gui.init()
gui.buildInstructions();
gui.buildReturnArrow();
gui.buildMenu();
gui.animate();
