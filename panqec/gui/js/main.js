import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/controls/OrbitControls.js';
import { OutlineEffect } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/effects/OutlineEffect.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/libs/dat.gui.module';

import { TopologicalCode } from './topologicalCode.js';

var defaultCode = codeDimension == 2 ? 'Toric 2D' : 'Toric 3D';
var defaultSize = codeDimension == 2 ? 6 : 4;

const params = {
    errorProbability: 0.1,
    L: defaultSize,
    noiseDeformationName: 'None',
    decoder: 'BP-OSD',
    max_bp_iter: 20,
    alpha: 0.4,
    beta: 0,
    channel_update: false,
    errorModel: 'Depolarizing',
    codeName: defaultCode,
    rotated: false,
    coprime: false,
    codeDeformationName: 'None'
};

let codeSize = {Lx: defaultSize, Ly: defaultSize, Lz: defaultSize};

const buttons = {
    'decode': decode,
    'addErrors': addRandomErrors
};

const COLORS = {
    background: 0x102542
}

const KEY_CODE = {'d': 68, 'r': 82, 'backspace': 8, 'o': 79, 'x': 88, 'z': 90}

let camera, controls, scene, renderer, effect, mouse, raycaster, intersects, menu;
let code;

init();
animate();

function init() {
    buildInstructions();
    buildReturnArrow();

    if (codeDimension == 2) {
        buildScene2D();
    }
    else {
        buildScene3D();
    }
    buildMenu();
    buildCode();

    if (codeDimension == 3) {
        controls.update();
    }
}

function buildScene2D() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color( COLORS.background );

    // Camera
    camera = new THREE.PerspectiveCamera( 10, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 25;
    camera.position.y = 0;
    camera.position.x = 0;

    const dirLight1 = new THREE.DirectionalLight( 0xffffff );
    dirLight1.position.set( 1, 1, 1 );
    scene.add( dirLight1 );

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    document.addEventListener("keydown", onDocumentKeyDown, false);
    document.addEventListener('mousedown', onDocumentMouseDown, false);
    window.addEventListener('resize', onWindowResize, false);
    window.addEventListener("contextmenu", e => e.preventDefault());
}

function buildScene3D() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

    camera.position.x = 5;
    camera.position.y = 3.5;
    camera.position.z = 7;

    const dirLight2 = new THREE.DirectionalLight( 0x002288 );
    dirLight2.position.set( - 1, - 1, - 1 );
    scene.add( dirLight2 );

    const dirLight3 = new THREE.DirectionalLight( 0x002288 );
    dirLight3.position.set(4, 4, 4);
    scene.add( dirLight3 );

    const pointLight = new THREE.PointLight( 0xffffff, 1, 0, 1);
    scene.add(pointLight);
    camera.add(pointLight);
    scene.add(camera);

    const ambientLight = new THREE.AmbientLight( 0x222222 );
    scene.add( ambientLight );

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer = new THREE.WebGLRenderer();
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    effect = new OutlineEffect(renderer);

    controls = new OrbitControls( camera, renderer.domElement );
    controls.maxPolarAngle = THREE.Math.degToRad(270);


    controls.update();

    document.addEventListener("keydown", onDocumentKeyDown, false);
    document.addEventListener( 'mousedown', onDocumentMouseDown, false );
    window.addEventListener('resize', onWindowResize, false);
}

async function buildCode() {
    let data = await getCodeData();
    let H = data['H'];
    let qubits = data['qubits'];
    let stabilizers = data['stabilizers'];
    let logical_z = data['logical_z'];
    let logical_x = data['logical_x'];
    let Lx = codeSize.Lx;
    let Ly = codeSize.Ly;
    let Lz = codeSize.Lz;

    if (codeDimension == 2) {
        var size = [Lx, Ly];
    }
    else {
        var size = [Lx, Ly, Lz]
    }

    code = new TopologicalCode(size, H, qubits, stabilizers);
    code.logical_x = logical_x;
    code.logical_z = logical_z;
    var maxCoordinates = code.build(scene);

    if (codeDimension == 2) {
        var fov = (camera.fov * Math.PI) / 180;
        camera.position.z = Math.max(maxCoordinates['x'], maxCoordinates['y']) / (2*Math.tan(fov/2)) + 10;
    }
}

async function changeLatticeSize() {
    codeSize.Lx = parseInt(params.L);
    codeSize.Ly = parseInt(params.L);
    codeSize.Lz = parseInt(params.L);

    if (params.coprime)
        codeSize.Lx += 1;

    code.qubits.forEach(q => {
        q.material.dispose();
        q.geometry.dispose();

        scene.remove(q);
    });

    code.stabilizers.forEach(s => {
        s.material.dispose();
        s.geometry.dispose();

        scene.remove(s);
    });

    await updateMenu();
    buildCode();
}

async function getCodeData() {
    let response = await fetch('/code-data', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({
            'Lx': codeSize.Lx,
            'Ly': codeSize.Ly,
            'Lz': codeSize.Lz,
            'code_name': params.codeName,
            'code_deformation_name': params.codeDeformationName,
            'rotated_picture': params.rotated
        })
    });

    let data  = await response.json();

    return data;
}

async function getCodeNames() {
    let response = await fetch('/code-names', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({'dimension': codeDimension})
    });

    let data  = await response.json();

    return data;
}

async function getDecoderNames() {
    let response = await fetch('/decoder-names', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({'code_name': params.codeName})
    });

    let data  = await response.json();

    return data;
}

async function getDeformationNames() {
    let response = await fetch('/deformation-names', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({'code_name': params.codeName})
    });

    let data  = await response.json();

    return data;
}

async function buildMenu() {
    menu = new GUI({width: 300});
    const codeFolder = menu.addFolder('Code')

    var codes = await getCodeNames();

    codeFolder.add(params, 'codeName', codes).name('Code type').onChange(changeLatticeSize);
    codeFolder.add(params, 'rotated').name('Rotated picture').onChange(changeLatticeSize);
    codeFolder.add(params, 'coprime').name('Coprime dimensions').onChange(changeLatticeSize);
    codeFolder.add(params, 'L', {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                                 '8': 8, '9': 9, '10':10, '11':11, '12': 12}).name('Lattice size').onChange(changeLatticeSize);
    codeFolder.open();

    updateMenu();
}

async function updateMenu() {
    // Clifford-deformation part
    var deformationNames = await getDeformationNames();
    deformationNames = ['None'].concat(deformationNames)

    if (!deformationNames.includes(params.codeDeformationName)) {
        params.codeDeformationName = 'None';
    }
    if (!deformationNames.includes(params.noiseDeformationName)) {
        params.noiseDeformationName = 'None';
    }

    var codeFolder = menu.__folders['Code'];

    codeFolder.__controllers.forEach(controller => {
        if (controller.property == 'codeDeformationName') {
            controller.remove();
        }
    });

    codeFolder.add(params, 'codeDeformationName', deformationNames).name('Clifford deformation').onChange(changeLatticeSize);

    if ('Error Model' in menu.__folders) {
        menu.removeFolder(menu.__folders['Error Model']);
    }

    const errorModelFolder = menu.addFolder('Error Model')
    errorModelFolder.add(params, 'errorModel',
        {'Pure X': 'Pure X', 'Pure Y': 'Pure Y', 'Pure Z': 'Pure Z', 'Depolarizing': 'Depolarizing'}
    ).name('Model');
    errorModelFolder.add(params, 'errorProbability', 0, 0.5).name('Probability');
    errorModelFolder.add(params, 'noiseDeformationName', deformationNames).name('Clifford deformation').onChange(changeLatticeSize);
    errorModelFolder.add(buttons, 'addErrors').name('▶ Add errors (r)');
    errorModelFolder.open();

    // Decoder part
    if ('Decoder' in menu.__folders) {
        menu.removeFolder(menu.__folders['Decoder']);
    }

    var decoders = await getDecoderNames();

    if (!decoders.includes(params.decoder)) {
        params.decoder = decoders[0];
    }

    const decoderFolder = menu.addFolder('Decoder');

    decoderFolder.add(params, 'decoder', decoders).name('Decoder');
    decoderFolder.add(params, 'max_bp_iter', 1, 1000, 1).name('Max iterations (BP)');
    decoderFolder.add(params, 'channel_update').name('Channel update (BP)');
    decoderFolder.add(params, 'alpha', 0.01, 2, 0.01).name('Alpha (MBP)');
    decoderFolder.add(params, 'beta', 0, 2, 0.01).name('Beta (MBP)');
    decoderFolder.add(buttons, 'decode').name("▶ Decode (d)");
    decoderFolder.open();
}

function toggleInstructions() {
    var closingCross = document.getElementById('closingCross');
    var instructions = document.getElementById('instructions');

    if (instructions.style.visibility == 'hidden') {
        instructions.style.visibility = 'visible';
        closingCross.innerHTML = "<a href='#'>× Instructions</a>";

    }
    else {
        instructions.style.visibility = 'hidden';
        closingCross.innerHTML = "<a href='#'>🔽 Instructions</a>";
    }
}

function buildInstructions() {
    var closingCross = document.createElement('div');
    closingCross.id = 'closingCross';
    closingCross.innerHTML = "<a href='#'>× Instructions</a>";
    closingCross.onclick = toggleInstructions;

    var instructions = document.createElement('div');
    instructions.id = 'instructions';
    if (codeDimension == 3) {
        instructions.innerHTML =
        "\
            <table style='border-spacing: 10px'>\
            <tr><td><b>Ctrl click</b></td><td>X error</td></tr>\
            <tr><td><b>Shift click</b></td><td>Z error</td></tr>\
            <tr><td><b>Backspace</b></td><td>Remove errors</td></tr>\
            <tr><td><b>R</b></td><td>Random errors</td></tr>\
            <tr><td><b>D</b></td><td>Decode</td></tr>\
            <tr><td><b>O</b></td><td>Toggle Opacity</td></tr>\
            <tr><td><b>Z</b></td><td>Logical Z</td></tr>\
            <tr><td><b>X</b></td><td>Logical X</td></tr>\
            </table>\
        ";
    }
    else {
        instructions.innerHTML =
        "\
            <table style='border-spacing: 10px'>\
            <tr><td><b>Ctrl click</b></td><td>X error</td></tr>\
            <tr><td><b>Shift click</b></td><td>Z error</td></tr>\
            <tr><td><b>Backspace</b></td><td>Remove errors</td></tr>\
            <tr><td><b>R</b></td><td>Random errors</td></tr>\
            <tr><td><b>D</b></td><td>Decode</td></tr>\
            <tr><td><b>O</b></td><td>Toggle Opacity</td></tr>\
            <tr><td><b>Z</b></td><td>Logical Z</td></tr>\
            <tr><td><b>X</b></td><td>Logical X</td></tr>\
            </table>\
        ";
    }
    document.body.appendChild(instructions);
    document.body.appendChild(closingCross);
}

function buildReturnArrow() {
    var returnArrow = document.createElement('div');
    returnArrow.id = 'returnArrow'
    returnArrow.innerHTML = "<a href='/'>❮</a>"

    document.body.appendChild(returnArrow);
}

function onDocumentMouseDown(event) {
    if (event.ctrlKey || event.shiftKey) {
        mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
        mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);

        intersects = raycaster.intersectObjects(code.qubits);
        if (intersects.length == 0) return;

        let selectedQubit = intersects[0].object;

        if (event.button == 0) {
            var x = selectedQubit.location[0]
            var y = selectedQubit.location[1]
            var z = selectedQubit.location[2]

            if (event.ctrlKey) {
                console.log('Selected qubit', selectedQubit.index, 'at', x, y, z);
                code.insertError(selectedQubit, 'X');
            }
            if (event.shiftKey) {
                console.log('Selected qubit', selectedQubit.index, 'at', x, y, z);
                code.insertError(selectedQubit, 'Z');
            }
        }
    }
}

async function getCorrection(syndrome) {
    let response = await fetch('/decode', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({
            'Lx': codeSize.Lx,
            'Ly': codeSize.Ly,
            'Lz': codeSize.Lz,
            'p': params.errorProbability,
            'max_bp_iter': params.max_bp_iter,
            'alpha': params.alpha,
            'beta': params.beta,
            'channel_update': params.channel_update,
            'syndrome': syndrome,
            'noise_deformation_name': params.noiseDeformationName,
            'decoder': params.decoder,
            'error_model': params.errorModel,
            'code_name': params.codeName,
            'code_deformation_name': params.codeDeformationName
        })
    });

    let data  = await response.json();

    return data
}

async function getRandomErrors() {
    let response = await fetch('/new-errors', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({
            'Lx': codeSize.Lx,
            'Ly': codeSize.Ly,
            'Lz': codeSize.Lz,
            'p': params.errorProbability,
            'noise_deformation_name': params.noiseDeformationName,
            'error_model': params.errorModel,
            'code_name': params.codeName,
            'code_deformation_name': params.codeDeformationName
        })
    });

    let data  = await response.json();

    return data;
}

async function addRandomErrors() {
    let errors = await getRandomErrors()
    let n = errors.length / 2;
    code.qubits.forEach((q, i) => {
        if (errors[i]) {
            code.insertError(q, 'X');
        }
        if (errors[n+i]) {
            code.insertError(q, 'Z');
        }
    });
}

function removeAllErrors() {
    code.qubits.forEach(q => {
        ['X', 'Z'].forEach(errorType => {
            if (q.hasError[errorType]) {
                code.insertError(q, errorType);
            }
        })
    });
}

async function decode() {
    let syndrome = code.getSyndrome();
    let correction = await getCorrection(syndrome)

    correction['x'].forEach((c,i) => {
        if(c) {
            code.insertError(code.qubits[i], 'X')
        }
    });
    correction['z'].forEach((c,i) => {
        if(c) {
            code.insertError(code.qubits[i], 'Z')
        }
    });
}

function onDocumentKeyDown(event) {
    var keyCode = event.which;

    if (keyCode == KEY_CODE['d']) {
        decode()
    }

    else if (keyCode == KEY_CODE['r']) {
        addRandomErrors();
    }

    else if (keyCode == KEY_CODE['backspace']) {
        removeAllErrors();
    }

    else if (keyCode == KEY_CODE['o']) {
        code.changeOpacity();
    }

    else if (keyCode == KEY_CODE['x']) {
        removeAllErrors();
        code.displayLogical(code.logical_x, 'X');
    }

    else if (keyCode == KEY_CODE['z']) {
        removeAllErrors();
        code.displayLogical(code.logical_z, 'Z');
    }
};

function onWindowResize(){

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}


function animate() {
    requestAnimationFrame(animate);

    // update the picking ray with the camera and mouse position
	raycaster.setFromCamera(mouse, camera);

    if (codeDimension == 3) {
        controls.update();
        effect.render(scene, camera);
    }
    else {
        renderer.render(scene, camera);
    }
}
