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
               errorX: 0xff0000, errorZ: 0x25CCF7, errorY: 0xa55eea, 
               activatedVertex: 0xf1c232, activatedFace: 0xf1c232}
const SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1}
const MIN_OPACITY = 0.1;
const MAX_OPACITY = 0.5;

const params = {
    opacity: MAX_OPACITY,
    errorProbability: 0.1,
    L: 2,
    deformed: false,
    decoder: 'bp'
};

let camera, controls, scene, renderer, effect, mouse, raycaster, intersects, gui;

let Hx, Hz;

let qubits = Array();
let vertices = Array();
let faces = Array();

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
    if (qubit.hasError[type]) {
        qubit.material.color.setHex(COLOR.edge);
        qubit.material.transparent = true;
    }
    else {
        qubit.material.color.setHex(COLOR.error);
        qubit.material.transparent = false;
    }
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
        qubit.material.color.setHex(COLOR.edge);
    }

    // Activate vertex stabilizers
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

    // Activate face stabilizers
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


async function buildCube() {
    let stabilizers = await getStabilizerMatrices();
    Hx = stabilizers['Hx'];
    Hz = stabilizers['Hz'];
    qubits = Array(Hx[0].length);
    vertices = Array(Hx.length);
    faces = Array(Hz.length)

    for(let x=0; x < params.L; x++) {
        for(let y=0; y < params.L; y++) {
            for(let z=0; z < params.L; z++) {
                buildVertex(x, y, z);

                AXES.forEach(axis => {
                    buildEdge(axis, x, y, z);
                    buildFace(axis, x, y, z)
                });
            }
        }
    }
}

function changeOpacity() {
    qubits.forEach(q => {
        if (!q.hasError) {
            q.material.opacity = params['opacity']
        }
    });

    vertices.forEach(v => {
        v.material.opacity = params['opacity']
    });
}

function changeLatticeSize() {
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

    buildCube();
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

    const material = new THREE.MeshToonMaterial({color: COLOR.vertex, opacity: params['opacity'], transparent: true});
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

    const material = new THREE.MeshPhongMaterial({color: COLOR.edge, opacity: params['opacity'], transparent: true});
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
            'L': params.L
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
            'deformed': params.deformed
        })
    });
    
    let data  = await response.json();

    return data;
}


function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x444488 );

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 5;
    camera.position.y = 2;
    camera.position.x = 2;

    const dirLight1 = new THREE.DirectionalLight( 0xffffff );
    dirLight1.position.set( 1, 1, 1 );
    scene.add( dirLight1 );

    const dirLight2 = new THREE.DirectionalLight( 0x002288 );
    dirLight2.position.set( - 1, - 1, - 1 );
    scene.add( dirLight2 );

    const ambientLight = new THREE.AmbientLight( 0x222222 );
    scene.add( ambientLight );

    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer = new THREE.WebGLRenderer();
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    effect = new OutlineEffect(renderer);
    
    controls = new OrbitControls( camera, renderer.domElement );

    buildCube();

    document.addEventListener("keydown", onDocumentKeyDown, false);
    document.addEventListener( 'mousedown', onDocumentMouseDown, false );
    window.addEventListener('resize', onWindowResize, false);

    gui = new GUI();
    gui.add(params, 'opacity', 0.1, 0.5).name('Opacity').onChange(changeOpacity);
    gui.add(params, 'errorProbability', 0, 0.5).name('Error probability');
    gui.add(params, 'L', 2, 10, 1).name('Lattice size').onChange(changeLatticeSize);
    gui.add(params, 'deformed').name('Deformed');
    gui.add(params, 'decoder', {'Belief Propagation': 'bp', 'SweepMatch': 'sweepmatch'});
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
    let syndrome_z = faces.map(f => + f.isActivated)
    let syndrome_x = vertices.map(v => + v.isActivated);
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
            'max_bp_iter': 10,
            'syndrome': syndrome,
            'deformed': params.deformed,
            'decoder': params.decoder
        })
    });
    
    let data  = await response.json();

    return data
}

async function onDocumentKeyDown(event) {
    var keyCode = event.which;

    if (keyCode == KEY_CODE['d']) {
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

    else if (keyCode == KEY_CODE['r']) {
        addRandomErrors();
    }

    else if (keyCode == KEY_CODE['backspace']) {
        removeAllErrors();
    }

    else if (keyCode == KEY_CODE['o']) {
        if (params['opacity'] == MIN_OPACITY) {
            params['opacity'] = MAX_OPACITY;
        }
        else {
            params['opacity'] = MIN_OPACITY;
        }
        gui.updateDisplay()
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

    effect.render( scene, camera );
}