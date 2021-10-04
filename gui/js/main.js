import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/controls/OrbitControls.js';
import { OutlineEffect } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/effects/OutlineEffect.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/libs/dat.gui.module';

import { ToricCode3D } from './codes/toric3d.js';
import { RhombicCode } from './codes/rhombic.js';

const MIN_OPACITY = 0.1;
const MAX_OPACITY = 0.6;

const params = {
    opacity: MAX_OPACITY,
    errorProbability: 0.1,
    L: 4,
    deformed: false,
    decoder: 'bp-osd-2',
    max_bp_iter: 10,
    errorModel: 'Depolarizing',
    codeName: 'rhombic'
};

const buttons = {
    'decode': decode,
    'addErrors': addRandomErrors
};

const KEY_CODE = {'d': 68, 'r': 82, 'backspace': 8, 'o': 79}

let camera, controls, scene, renderer, effect, mouse, raycaster, intersects, gui;
let code;

init();
animate();

function init() {
    buildInstructions();
    buildReturnArrow();
    buildScene();
    buildGUI();

    buildCode();

    controls.update();
}

function buildScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x444488);

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 6;
    camera.position.y = 3;
    camera.position.x = 3;

    let dirLight1 = new THREE.DirectionalLight( 0xffffff );
    dirLight1.position.set( 1, 1, 1 );
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

    controls.update();

    document.addEventListener("keydown", onDocumentKeyDown, false);
    document.addEventListener( 'mousedown', onDocumentMouseDown, false );
    window.addEventListener('resize', onWindowResize, false);
}

async function buildCode() {
    let stabilizers = await getStabilizerMatrices();
    let Hx = stabilizers['Hx'];
    let Hz = stabilizers['Hz'];
    let L = params.L;

    let codeClass = {'cubic': ToricCode3D,
                     'rhombic': RhombicCode}
                    //  'rotated': RotatedCode3D}

    code = new codeClass[params.codeName](L, Hx, Hz, scene);
    code.build();
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
            'L': params.L,
            'code_name': params.codeName
        })
    });
    
    let data  = await response.json();

    return data;
}

function buildGUI() {
    gui = new GUI();
    const codeFolder = gui.addFolder('Code')
    codeFolder.add(params, 'codeName', {'Cubic': 'cubic', 'Rhombic': 'rhombic', 'Rotated': 'rotated'}).name('Code type').onChange(changeLatticeSize);
    codeFolder.add(params, 'L', {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}).name('Lattice size').onChange(changeLatticeSize);

    const errorModelFolder = gui.addFolder('Error Model')
    errorModelFolder.add(params, 'errorModel', {'Pure X': 'Pure X', 'Pure Z': 'Pure Z', 'Depolarizing': 'Depolarizing'}).name('Model');
    errorModelFolder.add(params, 'errorProbability', 0, 0.5).name('Probability');
    errorModelFolder.add(params, 'deformed').name('Deformed');
    errorModelFolder.add(buttons, 'addErrors').name('▶ Add errors (r)');

    const decoderFolder = gui.addFolder('Decoder')
    decoderFolder.add(params, 'decoder', {'BP-OSD': 'bp-osd', 'BP-OSD-2': 'bp-osd-2', 'SweepMatch': 'sweepmatch'}).name('Decoder');
    decoderFolder.add(params, 'max_bp_iter', 1, 100, 1).name('Max iterations BP');
    decoderFolder.add(buttons, 'decode').name("▶ Decode (d)");
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
    if (event.ctrlKey) {
        mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
        mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        
        intersects = raycaster.intersectObjects(code.qubits);
        if (intersects.length == 0) return;
        
        let selectedQubit = intersects[0].object;
        
        switch (event.button) {
            case 0: // left click
                code.insertError(selectedQubit, 'X');
                break;
            case 2:
                code.insertError(selectedQubit, 'Z');
                break;
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
            'L': params.L,
            'p': params.errorProbability,
            'max_bp_iter': params.max_bp_iter,
            'syndrome': syndrome,
            'deformed': params.deformed,
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
            'L': params.L,
            'p': params.errorProbability,
            'deformed': params.deformed,
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

    controls.update()

    effect.render(scene, camera);
}