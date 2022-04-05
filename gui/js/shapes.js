import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

export {create_shape}

var create_shape = {'sphere': sphere,
                    'face': face,
                    'cylinder': cylinder}

function sphere(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.SphereGeometry(params['radius'], 32, 32);

    const material = new THREE.MeshToonMaterial({transparent: true});
    const sphere = new THREE.Mesh(geometry, material);

    sphere.position.x = x;
    sphere.position.y = y;
    sphere.position.z = z;

    return sphere;
}

function face(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.PlaneGeometry(params['w'], params['h']);

    const material = new THREE.MeshToonMaterial({transparent: true,
                                                 side: THREE.DoubleSide});
    const face = new THREE.Mesh(geometry, material);

    face.position.x = x;
    face.position.y = y;
    face.position.z = z;

    face.rotateZ(params['angle'])

    if (params['plane'] == 'yz')
        face.rotateX(Math.PI/2)
    else if (params['plane'] == 'xz')
        face.rotateY(Math.PI/2)

    return face
}

function cylinder(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.CylinderGeometry(params['radius'], params['radius'], params['length'], 32);

    const material = new THREE.MeshPhongMaterial({transparent: true});

    const qubit = new THREE.Mesh(geometry, material);

    qubit.position.x = x;
    qubit.position.y = y;
    qubit.position.z = z;

    qubit.rotateZ(params['angle']);

    if (params['axis'] == 'x') {
        qubit.rotateZ(Math.PI / 2);
    }
    else if (params['axis'] == 'z') {
        qubit.rotateX(Math.PI / 2);
    }

    return qubit;
}