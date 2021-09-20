import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/controls/OrbitControls.js';
import { OutlineEffect } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/effects/OutlineEffect.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/libs/dat.gui.module';

// Constants

const PI = Math.PI;
const KEY_CODE = {'d': 68, 'r': 82, 'backspace': 8, 'o': 79}

const X_AXIS = 1;
const Y_AXIS = 2;
const Z_AXIS = 0;
const AXES = [X_AXIS, Y_AXIS, Z_AXIS];
const X_ERROR = 0;
const Z_ERROR = 1;
const COLOR = {vertex: 0xf2f28c, face: 0xf2f28c, edge: 0xffbcbc, 
               cube: 0xf2f28c, triangle: 0xf2f2cc,
               errorX: 0xff0000, errorZ: 0x25CCF7, errorY: 0xa55eea, 
               activatedVertex: 0xf1c232, activatedFace: 0xf1c232, 
               activatedTriangle: 0xf1c232, activatedCube: 0xf1c232}
const SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1}
const MIN_OPACITY = 0.1;
const MAX_OPACITY = 0.5;

let currentOpacity = MAX_OPACITY;

const params = {
    opacity: MAX_OPACITY,
    errorProbability: 0.1,
    L: 4,
    deformed: false,
    decoder: 'bp',
    max_bp_iter: 10,
    errorModel: 'Depolarizing',
    codeName: 'cubic'
};

const buttons = {
    'decode': decode,
    'addErrors': addRandomErrors
};

let camera, controls, scene, renderer, effect, mouse, raycaster, intersects, gui;

let Hx, Hz;

let qubits = Array();
let vertices = Array();
let faces = Array();
let cubes = Array();
let triangles = Array();

init();
animate();

function getIndexQubit(axis, x, y, z) {
    let Lx = params.L;
    let Ly = params.L;
    let Lz = params.L;

    return axis*Lx*Ly*Lz + x*Ly*Lz + y*Lz + z;
}

function getIndexFace(axis, x, y, z) {
    let Lx = params.L;
    let Ly = params.L;
    let Lz = params.L;

    return axis*Lx*Ly*Lz + x*Ly*Lz + y*Lz + z;
}

function getIndexVertex(x, y, z) {
    let Lx = params.L;
    let Ly = params.L;
    let Lz = params.L;

    return x*Ly*Lz + y*Lz + z;
}

function getIndexCube(x, y, z) {
    let Lx = params.L;
    let Ly = params.L;
    let Lz = params.L;

    return Math.floor((x*Ly*Lz + y*Lz + z) / 2);
}

function getIndexTriangle(axis, x, y, z) {
    let Lx = params.L;
    let Ly = params.L;
    let Lz = params.L;

    return axis*Lx*Ly*Lz + x*Ly*Lz + y*Lz + z;
}

async function addRandomErrors() {
    let errors = await getRandomErrors()
    let n = errors.length / 2;
    qubits.forEach((q, i) => {
        if (errors[i]) {
            insertError(q, X_ERROR);
        }
        if (errors[n+i]) {
            insertError(q, Z_ERROR);
        }
    });
}

function removeAllErrors() {
    qubits.forEach(q => {
        [X_ERROR, Z_ERROR].forEach(errorType => {
            if (q.hasError[errorType]) {
                insertError(q, errorType);
            }
        })
    });
}

function insertError(qubit, type) {
    qubit.hasError[type] = !qubit.hasError[type];

    if (qubit.hasError[X_ERROR] || qubit.hasError[Z_ERROR]) {
        qubit.material.transparent = false;

        if (qubit.hasError[X_ERROR] && qubit.hasError[Z_ERROR]) {
            qubit.material.color.setHex(COLOR.errorY);
        }
        else if (qubit.hasError[X_ERROR]) {
            qubit.material.color.setHex(COLOR.errorX);
        }
        else {
            qubit.material.color.setHex(COLOR.errorZ);
        }
    }
    else {
        qubit.material.transparent = true;
        qubit.material.opacity = currentOpacity;
        qubit.material.color.setHex(COLOR.edge);
    }

    if (params.codeName == 'cubic') {
        updateVertices();
        updateFaces();
    }
    else if (params.codeName == 'rhombic') {
        updateTriangles();
        updateCubes();
    }
}

function updateVertices() {
    let nQubitErrors;
    for (let iVertex=0; iVertex < Hx.length; iVertex++) {
        nQubitErrors = 0
        for (let iQubit=0; iQubit < Hx[0].length; iQubit++) {
            if (Hx[iVertex][iQubit] == 1) {
                if (qubits[iQubit].hasError[X_ERROR]) {
                    nQubitErrors += 1
                }
            }
        }
        if (nQubitErrors % 2 == 1) {
            activateVertex(vertices[iVertex])
        }
        else {
            deactivateVertex(vertices[iVertex])
        }
    }
}

function updateFaces() {
    let nQubitErrors;
    for (let iFace=0; iFace < Hz.length; iFace++) {
        nQubitErrors = 0
        for (let iQubit=0; iQubit < Hx[0].length; iQubit++) {
            if (Hz[iFace][iQubit] == 1) {
                if (qubits[iQubit].hasError[Z_ERROR]) {
                    nQubitErrors += 1
                }
            }
        }
        if (nQubitErrors % 2 == 1) {
            activateFace(faces[iFace])
        }
        else {
            deactivateFace(faces[iFace])
        }
    }
}

function updateTriangles() {
    let nQubitErrors;
    for (let iTriangle=0; iTriangle < Hx.length; iTriangle++) {
        nQubitErrors = 0
        for (let iQubit=0; iQubit < Hx[0].length; iQubit++) {
            if (Hx[iTriangle][iQubit] == 1) {
                if (qubits[iQubit].hasError[X_ERROR]) {
                    nQubitErrors += 1
                }
            }
        }
        if (nQubitErrors % 2 == 1) {
            activateTriangle(triangles[iTriangle])
        }
        else {
            deactivateTriangle(triangles[iTriangle])
        }
    }
}

function updateCubes() {
    let nQubitErrors;
    for (let iCube=0; iCube < Hz.length; iCube++) {
        nQubitErrors = 0
        for (let iQubit=0; iQubit < Hx[0].length; iQubit++) {
            if (Hz[iCube][iQubit] == 1) {
                if (qubits[iQubit].hasError[Z_ERROR]) {
                    nQubitErrors += 1
                }
            }
        }
        if (nQubitErrors % 2 == 1) {
            activateCube(cubes[iCube])
        }
        else {
            deactivateCube(cubes[iCube])
        }
    }
}

function activateVertex(vertex) {
    vertex.isActivated = true;
    vertex.material.color.setHex(COLOR.activatedVertex);
    vertex.material.transparent = false;
}

function activateFace(face) {
    face.isActivated = true;
    face.material.color.setHex(COLOR.activatedFace);
    face.material.opacity = MAX_OPACITY;
}

function activateCube(cube) {
    cube.isActivated = true;
    cube.material.color.setHex(COLOR.activatedCube);
    cube.material.opacity = MAX_OPACITY;
}

function activateTriangle(triangle) {
    triangle.isActivated = true;
    triangle.material.color.setHex(COLOR.activatedTriangle);
    triangle.material.opacity = MAX_OPACITY;
    triangle.material.transparent = false;
}

function deactivateVertex(vertex) {
    vertex.isActivated = false;
    vertex.material.color.setHex(COLOR.vertex);
    vertex.material.transparent = true;
}

function deactivateFace(face) {
    face.isActivated = false;
    face.material.color.setHex(COLOR.face);
    face.material.opacity = 0;
}

function deactivateCube(cube) {
    cube.isActivated = false;
    cube.material.color.setHex(COLOR.cube);
    cube.material.opacity = currentOpacity;
}

function deactivateTriangle(triangle) {
    triangle.isActivated = false;
    triangle.material.color.setHex(COLOR.triangle);
    triangle.material.opacity = currentOpacity;
    triangle.material.transparent = true;
}


async function buildCode() {
    let stabilizers = await getStabilizerMatrices();
    Hx = stabilizers['Hx'];
    Hz = stabilizers['Hz'];
    var logical_xs = stabilizers['logical_xs']
    var logical_zs = stabilizers['logical_zs']

    qubits = Array(Hx[0].length);

    vertices = Array(Hx.length);
    faces = Array(Hz.length)

    triangles = Array(Hx.length)
    cubes = Array(Hz.length)

    for(let x=0; x < params.L; x++) {
        for(let y=0; y < params.L; y++) {
            for(let z=0; z < params.L; z++) {
                for (let axis=0; axis < 3; axis++) {
                    buildEdge(axis, x, y, z);
                }
                if (params.codeName == 'cubic') {
                    buildVertex(x, y, z);

                    for (let axis=0; axis < 3; axis++) {
                        buildFace(axis, x, y, z);
                    }
                }
                else if (params.codeName == 'rhombic') {
                    buildCube(x, y, z);

                    for (let axis=0; axis < 4; axis++) {
                        buildTriangle(axis, x, y, z);
                    }
                }

            }
        }
    }

    // qubits.forEach((q, i) => {
    //     if (logical_zs[2][i]) {
    //         insertError(q, Z_ERROR);
    //     }
    // });
}

function changeOpacity() {
    if (currentOpacity == MIN_OPACITY) {
        currentOpacity = MAX_OPACITY;
    }
    else {
        currentOpacity = MIN_OPACITY;
    }

    qubits.forEach(q => {
        if (!q.hasError[X_ERROR] && !q.hasError[Z_ERROR]) {
            q.material.opacity = currentOpacity;
        }
    });

    vertices.forEach(v => {
        v.material.opacity = currentOpacity;
    });

    cubes.forEach(c => {
        if (!c.isActivated) {
            c.material.opacity = currentOpacity;
            c.children[0].material.opacity = currentOpacity;
        }
    });

    triangles.forEach(t => {
        if (!t.isActivated) {
            t.material.opacity = currentOpacity;
            t.children[0].material.opacity = currentOpacity;
        }
    });
}

function changeLatticeSize() {
    params.L = parseInt(params.L)
    qubits.forEach(q => {
        q.material.dispose();
        q.geometry.dispose();

        scene.remove(q);
    });

    vertices.forEach(v => {
        v.material.dispose();
        v.geometry.dispose();

        scene.remove(v);
    });

    faces.forEach(f => {
        f.material.dispose();
        f.geometry.dispose();

        scene.remove(f);
    });

    cubes.forEach(c => {
        c.material.dispose();
        c.geometry.dispose();

        scene.remove(c);
    });

    triangles.forEach(t => {
        t.material.dispose();
        t.geometry.dispose();

        scene.remove(t);
    });


    buildCode();
}

function buildTriangle(axis, x, y, z) {
    const L = SIZE.lengthEdge / 4

    const geometry = new THREE.BufferGeometry();

    if (axis == 0) {
        if ((x + y + z) % 2 == 0) {
            var vertices = new Float32Array([
                x+L,   y,   z,
                x,   y+L, z,
                x,   y,   z+L
            ]);
        }
        else {
            var vertices = new Float32Array([
                x-L,   y,   z,
                x,   y-L, z,
                x,   y,   z-L
            ]);
        }
    }

    else if (axis == 1) {
        if ((x + y + z) % 2 == 0) {
            var vertices = new Float32Array([
                x+L,   y,   z,
                x,   y-L, z,
                x,   y,   z-L
            ]);
        }
        else {
            var vertices = new Float32Array([
                x-L,   y,   z,
                x,   y+L, z,
                x,   y,   z+L
            ]);
        }
    }

    else if (axis == 2) {
        if ((x + y + z) % 2 == 0) {
            var vertices = new Float32Array([
                x-L,   y,   z,
                x,   y+L, z,
                x,   y,   z-L
            ]);
        }
        else {
            var vertices = new Float32Array([
                x+L,   y,   z,
                x,   y-L, z,
                x,   y,   z+L
            ]);
        }
    }

    else if (axis == 3) {
        if ((x + y + z) % 2 == 0) {
            var vertices = new Float32Array([
                x-L,   y,   z,
                x,   y-L, z,
                x,   y,   z+L
            ]);
        }
        else {
            var vertices = new Float32Array([
                x+L,   y,   z,
                x,   y+L, z,
                x,   y,   z-L
            ]);
        }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3) );

    const material = new THREE.MeshBasicMaterial({color: COLOR.triangle, opacity: currentOpacity, transparent: true, side: THREE.DoubleSide});
    const triangle = new THREE.Mesh(geometry, material);

    var geo = new THREE.EdgesGeometry(triangle.geometry);
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 1, opacity: currentOpacity, transparent: true});
    var wireframe = new THREE.LineSegments(geo, mat);
    wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
    triangle.add(wireframe);

    let index = getIndexTriangle(axis, x, y, z);
    triangle.index = index;
    triangle.isActivated = false;
    triangles[index] = triangle;

    scene.add(triangle);
}

function buildCube(x, y, z) {
    if ((x + y + z) % 2 == 1) {
        const L = SIZE.lengthEdge - 0.3
        const geometry = new THREE.BoxBufferGeometry(L, L, L);
        const material = new THREE.MeshToonMaterial({color: COLOR.cube, opacity: currentOpacity, transparent: true});
        const cube = new THREE.Mesh(geometry, material);

        var geo = new THREE.EdgesGeometry( cube.geometry );
        var mat = new THREE.LineBasicMaterial( { color: 0x000000, linewidth: 2, opacity: currentOpacity, transparent: true } );
        var wireframe = new THREE.LineSegments( geo, mat );
        wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
        cube.add(wireframe);

        cube.position.x = x + SIZE.lengthEdge / 2;
        cube.position.y = y + SIZE.lengthEdge / 2;
        cube.position.z = z + SIZE.lengthEdge / 2;

        let index = getIndexCube(x, y, z);
        cube.index = index;
        cube.isActivated = false;
        cubes[index] = cube;

        scene.add(cube);
    }
}

function buildFace(axis, x, y, z) {
    const geometry = new THREE.PlaneGeometry(SIZE.lengthEdge-0.3, SIZE.lengthEdge-0.3);

    const material = new THREE.MeshToonMaterial({color: COLOR.face, opacity: 0, transparent: true, side: THREE.DoubleSide});
    const face = new THREE.Mesh(geometry, material);

    face.position.x = x;
    face.position.y = y;
    face.position.z = z;

    if (axis == Y_AXIS) {
        face.position.x += SIZE.lengthEdge / 2;
        face.position.y += SIZE.lengthEdge / 2;
    }
    if (axis == X_AXIS) {
        face.position.x += SIZE.lengthEdge / 2;
        face.position.z += SIZE.lengthEdge / 2;
        face.rotateX(PI / 2)
    }
    else if (axis == Z_AXIS) {
        face.position.z += SIZE.lengthEdge / 2
        face.position.y += SIZE.lengthEdge / 2
        face.rotateY(PI / 2)
    }

    let index = getIndexFace(axis, x, y, z);

    face.index = index;
    face.isActivated = false;

    faces[index] = face;

    scene.add(face);
}

function buildVertex(x, y, z) {
    const geometry = new THREE.SphereGeometry(SIZE.radiusVertex, 32, 32);

    const material = new THREE.MeshToonMaterial({color: COLOR.vertex, opacity: currentOpacity, transparent: true});
    const sphere = new THREE.Mesh(geometry, material);

    sphere.position.x = x;
    sphere.position.y = y;
    sphere.position.z = z;

    let index = getIndexVertex(x, y, z);

    sphere.index = index;
    sphere.isActivated = false;

    vertices[index] = sphere;

    scene.add(sphere);

    // Vertex number

    // const loader = new THREE.FontLoader();

    // loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function ( font ) {

    //     const geometryText = new THREE.TextGeometry(String(index), {
    //         font: font,
    //         size: 0.1,
    //         height: 0.03,
    //         curveSegments: 2,
    //         bevelEnabled: true,
    //         bevelThickness: 0.001,
    //         bevelSize: 0.001,
    //         bevelSegments: 3
    //     } );

    //     let materialText = new THREE.MeshPhongMaterial({color: 0x00ff00});
    //     let text = new THREE.Mesh(geometryText, materialText);

    //     text.position.x = x;
    //     text.position.y = y;
    //     text.position.z = z;

    //     scene.add(text)
    // } );
}

function buildEdge(axis, x, y, z) {
    const geometry = new THREE.CylinderGeometry(SIZE.radiusEdge, SIZE.radiusEdge, SIZE.lengthEdge, 32);

    const material = new THREE.MeshPhongMaterial({color: COLOR.edge, opacity: currentOpacity, transparent: true});
    const edge = new THREE.Mesh(geometry, material);

    edge.position.x = x;
    edge.position.y = y;
    edge.position.z = z;

    if (axis == X_AXIS) {
        edge.position.y += SIZE.lengthEdge / 2
    }
    if (axis == Y_AXIS) {
        edge.rotateX(PI / 2)
        edge.position.z += SIZE.lengthEdge / 2
    }
    else if (axis == Z_AXIS) {
        edge.rotateZ(PI / 2)
        edge.position.x += SIZE.lengthEdge / 2
    }

    edge.hasError = [false, false];

    let index = getIndexQubit(axis, x, y, z)

    edge.index = index;
    qubits[index] = edge;

    scene.add(edge);

    // Text for qubit number

    // const loader = new THREE.FontLoader();

    // loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function ( font ) {

    //     const geometryText = new THREE.TextGeometry(String(index), {
    //         font: font,
    //         size: 0.1,
    //         height: 0.03,
    //         curveSegments: 2,
    //         bevelEnabled: true,
    //         bevelThickness: 0.001,
    //         bevelSize: 0.001,
    //         bevelSegments: 3
    //     } );

    //     let materialText = new THREE.MeshPhongMaterial({color: 0x0});
    //     let text = new THREE.Mesh(geometryText, materialText);

    //     text.position.x = x;
    //     text.position.y = y;
    //     text.position.z = z;

    //     if (axis == X_AXIS) {
    //         text.position.y -= SIZE.lengthEdge / 2
    //     }
    //     if (axis == Y_AXIS) {
    //         text.rotateX(PI / 2)
    //         text.position.z -= SIZE.lengthEdge / 2
    //     }
    //     else if (axis == Z_AXIS) {
    //         text.rotateZ(PI / 2)
    //         text.position.x -= SIZE.lengthEdge / 2
    //     }

    //     scene.add(text)
    // } );

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
            'error_model': params.errorModel
        })
    });
    
    let data  = await response.json();

    return data;
}

function toggleInstructions() {
    var closingCross = document.getElementById('closingCross');
    var instructions = document.getElementById('instructions');

    if (instructions.style.visibility == 'hidden') {
        instructions.style.visibility = 'visible';
        closingCross.innerHTML = "<a style='color: inherit; text-decoration: inherit' href='#'>× Instructions</a>";

    }
    else {
        instructions.style.visibility = 'hidden';
        closingCross.innerHTML = "<a style='color: inherit; text-decoration: inherit' href='#'>🔽 Instructions</a>";
    }
}

function buildGUI() {
    gui = new GUI();
    const codeFolder = gui.addFolder('Code')
    codeFolder.add(params, 'codeName', {'Cubic': 'cubic', 'Rhombic': 'rhombic'}).name('Code type').onChange(changeLatticeSize);
    codeFolder.add(params, 'L', {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}).name('Lattice size').onChange(changeLatticeSize);

    const errorModelFolder = gui.addFolder('Error Model')
    errorModelFolder.add(params, 'errorModel', {'Pure X': 'Pure X', 'Pure Z': 'Pure Z', 'Depolarizing': 'Depolarizing'}).name('Model');
    errorModelFolder.add(params, 'errorProbability', 0, 0.5).name('Probability');
    errorModelFolder.add(params, 'deformed').name('Deformed');
    errorModelFolder.add(buttons, 'addErrors').name('▶ Add errors (r)');

    const decoderFolder = gui.addFolder('Decoder')
    decoderFolder.add(params, 'decoder', {'Belief Propagation': 'bp', 'SweepMatch': 'sweepmatch'}).name('Decoder');
    decoderFolder.add(params, 'max_bp_iter', 1, 100, 1).name('Max iterations BP');
    decoderFolder.add(buttons, 'decode').name("▶ Decode (d)");
}

function buildInstructions() {
    var closingCross = document.createElement('div');
    closingCross.id = 'closingCross';
    closingCross.style.position = 'absolute';
    closingCross.style.top = 15 + 'px';
    closingCross.style.left = 15 + 'px';
    closingCross.style.fontFamily = 'sans-serif';
    closingCross.style.fontSize = 20 + 'px';
    closingCross.style.color = 'white';
    closingCross.innerHTML = "<a style='color: inherit; text-decoration: inherit' href='#'>× Instructions</a>";
    closingCross.addEventListener('mouseenter', e => { closingCross.style.opacity = 0.6; })
    closingCross.addEventListener('mouseleave', e => { closingCross.style.opacity = 1; })
    closingCross.onclick = toggleInstructions;

    var instructions = document.createElement('div');
    instructions.id = 'instructions';
    instructions.style.position = 'absolute';
    instructions.style.top = 40 + 'px';
    instructions.style.left = 40 + 'px';
    instructions.style.fontFamily = 'sans-serif'
    instructions.style.fontSize = 20 + 'px'
    instructions.style.color = "white";
    instructions.style.opacity = 0.6;
    instructions.style.transparent = "true"
    instructions.innerHTML = 
    "\
        <table style='border-spacing: 10px'>\
        <tr><td><b>CTRL + Left click</b></td><td>X error</td></tr>\
        <tr><td><b>CTRL + Right click</b></td><td>Z error</td></tr>\
        <tr><td><b>Backspace</b></td><td>Remove errors</td></tr>\
        <tr><td><b>R</b></td><td>Random errors</td></tr>\
        <tr><td><b>O</b></td><td>Toggle opacity</td></tr>\
        <tr><td><b>D</b></td><td>Decode</td></tr>\
        </table>\
    ";
    document.body.appendChild(instructions);
    document.body.appendChild(closingCross);
}


function init() {
    // Display instructions
    buildInstructions()


    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x444488 );

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 6;
    camera.position.y = 3;
    camera.position.x = 3;

    const dirLight1 = new THREE.DirectionalLight( 0xffffff );
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
    // effect = renderer;
    
    controls = new OrbitControls( camera, renderer.domElement );

    buildCode();

    document.addEventListener("keydown", onDocumentKeyDown, false);
    document.addEventListener( 'mousedown', onDocumentMouseDown, false );
    window.addEventListener('resize', onWindowResize, false);

    buildGUI()

    controls.update();
}

function onDocumentMouseDown(event) {
    if (event.ctrlKey) {
        mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
        mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        
        intersects = raycaster.intersectObjects(qubits);
        if (intersects.length == 0) return;
        
        let selectedQubit = intersects[0].object;
        
        switch (event.button) {
            case 0: // left click
                insertError(selectedQubit, X_ERROR);
                break;
            case 2:
                insertError(selectedQubit, Z_ERROR);
                break;
        }
    }
}

function getSyndrome() {
    let syndrome_z, syndrome_x;
    if (params.codeName == 'cubic') {
        syndrome_z = faces.map(f => + f.isActivated)
        syndrome_x = vertices.map(v => + v.isActivated);
    }
    else if (params.codeName == 'rhombic') {
        syndrome_z = cubes.map(c => + c.isActivated)
        syndrome_x = triangles.map(t => + t.isActivated);
    }
        
    return syndrome_z.concat(syndrome_x)
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

async function decode() {
    let syndrome = getSyndrome();
    let correction = await getCorrection(syndrome)

    correction['x'].forEach((c,i) => {
        if(c) {
            insertError(qubits[i], X_ERROR)
        }
    });
    correction['z'].forEach((c,i) => {
        if(c) {
            insertError(qubits[i], Z_ERROR)
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
        changeOpacity();
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