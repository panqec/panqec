import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/controls/OrbitControls.js';
import { OutlineEffect } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/effects/OutlineEffect.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/libs/dat.gui.module';

import { ToricCode2D } from './codes/toric2d.js';
import { ToricCode3D } from './codes/toric3d.js';
import { RhombicCode } from './codes/rhombic.js';
import { RotatedPlanarCode3D } from './codes/rotatedPlanar3d.js';
import { RotatedToricCode3D } from './codes/rotatedToric3d.js';
import { RpRotatedPlanarCode3D } from './codes/rpRotatedPlanar3d.js';
import { PlanarCode3D } from './codes/planar3d.js';


const MIN_OPACITY = 0.2;
const MAX_OPACITY = 0.6;

var defaultCode = codeDimension == 2 ? 'toric-2d' : 'rp-rotated-planar-3d';

const params = {
    opacity: MAX_OPACITY,
    errorProbability: 0.1,
    L: 2,
    deformation: "None",
    decoder: 'bp-osd-2',
    max_bp_iter: 10,
    errorModel: 'Depolarizing',
    codeName: defaultCode
};

const buttons = {
    'decode': decode,
    'addErrors': addRandomErrors
};

const KEY_CODE = {'d': 68, 'r': 82, 'backspace': 8, 'o': 79, 'x': 88, 'z': 90}

let camera, controls, scene, renderer, effect, mouse, raycaster, intersects, gui;
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
    buildGUI();
    buildCode();

    if (codeDimension == 3) {
        controls.update();
    }
}

function buildScene2D() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x444488 );

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
    scene.background = new THREE.Color(0x444488);

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    let radius = 4;
    let theta = 3 + 1.4;
    let phi = Math.PI/2;
    camera.position.z = radius * Math.cos(theta);
    camera.position.y = radius * Math.sin(phi) * Math.sin(theta);
    camera.position.x = radius * Math.cos(phi) * Math.sin(theta);


    let dirLight1 = new THREE.DirectionalLight( 0xffffff );
    dirLight1.position.set( radius * Math.cos(theta), radius * Math.sin(phi) * Math.sin(theta), radius * Math.cos(phi) * Math.sin(theta));
    scene.add( dirLight1 );

    const dirLight2 = new THREE.DirectionalLight( 0x002288 );
    dirLight2.position.set( - 1, - 1, - 1 );
    scene.add( dirLight2 );

    const dirLight3 = new THREE.DirectionalLight( 0x002288 );
    dirLight3.position.set(4, 4, 4);
    scene.add( dirLight3 );

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
    let stabilizers = await getStabilizerMatrices();
    let Hx = stabilizers['Hx'];
    let Hz = stabilizers['Hz'];
    let indices = stabilizers['indices'];
    let logical_z = stabilizers['logical_z'];
    let logical_x = stabilizers['logical_x'];
    let Lx = params.L;
    let Ly = params.L;
    let Lz = params.L;

    if (codeDimension == 2) {
        var size = [Lx, Ly];
    }
    else {
        var size = [Lx, Ly, Lz]
    }

    let codeClass = {'toric-2d': ToricCode2D,
                     'toric-3d': ToricCode3D,
                     'planar-3d': PlanarCode3D,
                     'rhombic': RhombicCode,
                     'rotated-planar-3d': RotatedPlanarCode3D,
                     'rotated-toric-3d': RotatedToricCode3D,
                     'rp-rotated-planar-3d': RpRotatedPlanarCode3D}

    code = new codeClass[params.codeName](size, Hx, Hz, indices, scene);
    code.logical_x = logical_x;
    code.logical_z = logical_z;
    code.build();
    // code.displayLogical(logical_z, 'Z', 0);
}

function changeLatticeSize() {
    params.L = parseInt(params.L)
    code.qubits.forEach(q => {
        q.material.dispose();
        q.geometry.dispose();

        scene.remove(q);
    });

    ['X', 'Z'].forEach(pauli => {
        code.stabilizers[pauli].forEach(s => {
            s.material.dispose();
            s.geometry.dispose();

            scene.remove(s);
        });
    });

    buildCode();
}

async function getStabilizerMatrices() {
    let response = await fetch('/stabilizer-matrix', {
        headers: {
            'Content-Type': 'application/json'
          },
        method: 'POST',
        body: JSON.stringify({
            'Lx': params.L,
            'Ly': params.L,
            'Lz': params.L,
            'code_name': params.codeName
        })
    });

    let data  = await response.json();

    return data;
}

function buildGUI() {
    gui = new GUI();
    const codeFolder = gui.addFolder('Code')

    var codes2d = {'Toric': 'toric-2d'};
    var codes3d = {'Toric 3D': 'toric-3d', 'Rotated Toric 3D': 'rotated-toric-3d',
                   'Rhombic': 'rhombic',
                   'Planar 3D': 'planar-3d', 'Rotated Planar 3D': 'rotated-planar-3d', 'RP Rotated Planar 3D': 'rp-rotated-planar-3d'};

    var codes = codeDimension == 2 ? codes2d : codes3d;

    codeFolder.add(params, 'codeName', codes).name('Code type').onChange(changeLatticeSize);
    codeFolder.add(params, 'L', {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}).name('Lattice size').onChange(changeLatticeSize);
    codeFolder.open();

    const errorModelFolder = gui.addFolder('Error Model')
    errorModelFolder.add(params, 'errorModel', {'Pure X': 'Pure X', 'Pure Z': 'Pure Z', 'Depolarizing': 'Depolarizing'}).name('Model');
    errorModelFolder.add(params, 'errorProbability', 0, 0.5).name('Probability');
    errorModelFolder.add(params, 'deformation', {'None': 'None', 'XZZX': 'XZZX', 'XY': 'XY', 'Rhombic': 'Rhombic'}).name('Deformation');
    errorModelFolder.add(buttons, 'addErrors').name('▶ Add errors (r)');
    errorModelFolder.open();

    const decoderFolder = gui.addFolder('Decoder')
    decoderFolder.add(params, 'decoder', {
        'BP-OSD': 'bp-osd', 'BP-OSD-2': 'bp-osd-2',
        'SweepMatch': 'sweepmatch', 'InfZOpt': 'infzopt'
    }).name('Decoder');
    decoderFolder.add(params, 'max_bp_iter', 1, 100, 1).name('Max iterations BP');
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
    instructions.innerHTML =
    "\
        <table style='border-spacing: 10px'>\
        <tr><td><b>Ctrl-left click</b></td><td>X error</td></tr>\
        <tr><td><b>Ctrl-right click</b></td><td>Z error</td></tr>\
        <tr><td><b>Backspace</b></td><td>Remove errors</td></tr>\
        <tr><td><b>R</b></td><td>Random errors</td></tr>\
        <tr><td><b>D</b></td><td>Decode</td></tr>\
        <tr><td><b>O</b></td><td>Toggle Opacity</td></tr>\
        <tr><td><b>Z</b></td><td>Logical Z</td></tr>\
        <tr><td><b>X</b></td><td>Logical X</td></tr>\
        </table>\
    ";
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

        if (event.ctrlKey) {
            switch (event.button) {
                case 0: // left click
                    code.insertError(selectedQubit, 'X');
                    break;
                case 2:
                    code.insertError(selectedQubit, 'Z');
                    break;
            }
        } else {
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
            'Lx': params.L,
            'Ly': params.L,
            'Lz': params.L,
            'p': params.errorProbability,
            'max_bp_iter': params.max_bp_iter,
            'syndrome': syndrome,
            'deformation': params.deformation,
            'decoder': params.decoder,
            'error_model': params.errorModel,
            'code_name': params.codeName
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
            'Lx': params.L,
            'Ly': params.L,
            'Lz': params.L,
            'p': params.errorProbability,
            'deformation': params.deformation,
            'error_model': params.errorModel,
            'code_name': params.codeName
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
        code.displayLogical(code.logical_x, 'X', 0);
    }

    else if (keyCode == KEY_CODE['z']) {
        code.displayLogical(code.logical_z, 'Z', 0);
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
