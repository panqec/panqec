import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/controls/OrbitControls.js';
import { OutlineEffect } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/effects/OutlineEffect.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/libs/dat.gui.module';

// Constants

const PI = Math.PI;
const KEY_CODE = {'d': 68, 'r': 82, 'backspace': 8, 'o': 79}

const X_AXIS = 0;
const Y_AXIS = 1;
const Z_AXIS = 2;
const COLOR = {vertex: 0xf2f28c, edge: 0xffbcbc, error: 0xff0000}
const SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1}
const MIN_OPACITY = 0.1;
const MAX_OPACITY = 0.5;

const params = {
    opacity: MAX_OPACITY,
    errorProbability: 0.1,
};


// Get data from servers

// var socket = socketIo.io.connect('http://' + document.domain + ':' + location.port);
// socket.on('connect', function() {
//      console.log('connected');
// });
// socket.on('message', function(data) {
//      console.log(data);
// });

// Start the interface

let camera, controls, scene, renderer, effect, mouse, raycaster, intersects, gui;

let Lx = 5;
let Ly = 5;
let Lz = 5;

let qubits = Array();
let vertices = Array();

init();
animate();

function getIndexQubit(axis, x, y, z) {
    return axis*3*Lx*Ly*Lz + x*Ly*Lz + y*Lz + z;
}

function getIndexVertex(x, y, z) {
    return x*Ly*Lz + y*Lz + z;
}

function add_random_errors(p) {
    qubits.forEach(q => {
        if (Math.random() < p) {
            insert_error(q);
        }
    });
}

function remove_all_errors() {
    qubits.forEach(q => {
        if (q.hasError) {
            insert_error(q);
        }
    });
}

function insert_error(qubit) {
    if (qubit.hasError) {
        qubit.material.color.setHex(COLOR.edge);
        qubit.material.transparent = true;
    }
    else {
        qubit.material.color.setHex(COLOR.error);
        qubit.material.transparent = false;
    }
    qubit.hasError = !qubit.hasError;
}


function buildCube(Lx, Ly, Lz) {
    for(let x=0; x < Lx; x++) {
        for(let y=0; y < Ly; y++) {
            for(let z=0; z < Lz; z++) {
                buildVertex(x, y, z);
                buildEdge(X_AXIS, x, y, z);
                buildEdge(Y_AXIS, x, y, z);
                buildEdge(Z_AXIS, x, y, z);
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

function buildVertex(x, y, z) {
    const geometry = new THREE.SphereGeometry(SIZE.radiusVertex, 32, 32);

    const alpha = 0.5;
    const beta = 0.5;
    const gamma = 0.5;
    const diffuseColor = new THREE.Color().setHSL( alpha, 0.5, gamma * 0.5 + 0.1 ).multiplyScalar( 1 - beta * 0.2 );

    const material = new THREE.MeshToonMaterial({color: COLOR.vertex, opacity: params['opacity'], transparent: true});
    const sphere = new THREE.Mesh(geometry, material);

    sphere.position.x = x;
    sphere.position.y = y;
    sphere.position.z = z;

    vertices.push(sphere)

    // sphere.adjacentQubits = 

    scene.add(sphere);
    // vertices.push(sphere)
}

function buildEdge(axis, x, y, z) {
    const geometry = new THREE.CylinderGeometry(SIZE.radiusEdge, SIZE.radiusEdge, SIZE.lengthEdge, 32);

    const material = new THREE.MeshPhongMaterial({color: COLOR.edge, opacity: params['opacity'], transparent: true});
    const edge = new THREE.Mesh(geometry, material);

    edge.position.x = x;
    edge.position.y = y;
    edge.position.z = z;

    if (axis == X_AXIS) {
        edge.position.y -= SIZE.lengthEdge / 2
    }
    if (axis == Y_AXIS) {
        edge.rotateX(PI / 2)
        edge.position.z -= SIZE.lengthEdge / 2
    }
    else if (axis == Z_AXIS) {
        edge.rotateZ(PI / 2)
        edge.position.x -= SIZE.lengthEdge / 2
    }

    edge.hasError = false;

    qubits.push(edge)
    scene.add(edge);
}

function getStabilizerMatrix() {
    fetch('/stabilizer-matrix').then(response => response.json()).then(data => console.log(data))
}


function init() {
    getStabilizerMatrix();

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x444488 );

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 10;
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

    buildCube(5, 5, 5);

    document.addEventListener("keydown", onDocumentKeyDown, false);
    document.addEventListener( 'mousedown', onDocumentMouseDown, false );
    window.addEventListener('resize', onWindowResize, false);

    gui = new GUI();
    gui.add(params, 'opacity', 0.1, 0.5).name('Opacity').onChange(changeOpacity);
    gui.add(params, 'errorProbability', 0, 0.5).name('Error probability');
    
    controls.update();
}

function onDocumentMouseDown(event) 
{
    if (event.ctrlKey) {
        console.log("Click.");
        
        mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
        mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
    
        intersects = raycaster.intersectObjects(qubits);
        if (intersects.length == 0) return;
        
        const selectedQubit = intersects[0].object;
        
        insert_error(selectedQubit);
    }

}

function onDocumentKeyDown(event) {
    var keyCode = event.which;

    if (keyCode == KEY_CODE['d']) {
        var xhr = new XMLHttpRequest();
        xhr.open("POST", '/decode', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({
            size: [2, 2, 2],
            syndrome: [0, 0, 0],
            p: params['errorProbability'],
            max_bp_iter: 10
        }));
    }

    else if (keyCode == KEY_CODE['r']) {
        add_random_errors(params['errorProbability']);
    }

    else if (keyCode == KEY_CODE['backspace']) {
        remove_all_errors();
    }

    else if (keyCode == KEY_CODE['backspace']) {
        remove_all_errors();
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