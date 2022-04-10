import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

export {create_shape}

var create_shape = {'sphere': sphere,
                    'rectangle': rectangle,
                    'cylinder': cylinder,
                    'octahedron': octahedron,
                    'cube': cube,
                    'triangle': triangle}

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

function rectangle(location, params) {
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

    var normal = {'x': params['normal'][0], 'y': params['normal'][1], 'z': params['normal'][2]};
    var norm = Math.sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    var norm_xz = Math.sqrt(normal.x*normal.x + normal.z*normal.z);

    var theta = (norm_xz == 0) ? 0 : Math.acos(normal.x / norm_xz);
    var alpha = Math.acos(norm_xz / norm);

    face.rotateY(theta + Math.PI / 2);
    face.rotateX(alpha);
    face.rotateZ(params['angle']);

    return face
}

function cylinder(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.CylinderGeometry(params['radius'], params['radius'], params['length'], 32);

    const material = new THREE.MeshPhongMaterial({transparent: true});

    const cylinder = new THREE.Mesh(geometry, material);

    cylinder.position.x = x;
    cylinder.position.y = y;
    cylinder.position.z = z;

    cylinder.rotateZ(params['angle']);

    if (params['axis'] == 'x') {
        cylinder.rotateZ(Math.PI / 2);
    }
    else if (params['axis'] == 'z') {
        cylinder.rotateX(Math.PI / 2);
    }

    return cylinder;
}

function octahedron(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.OctahedronGeometry(params['length']);

    const material = new THREE.MeshToonMaterial({transparent: true,
                                                 side: THREE.DoubleSide});
    const octahedron = new THREE.Mesh(geometry, material);

    octahedron.position.x = x;
    octahedron.position.y = y;
    octahedron.position.z = z;

    octahedron.rotateZ(params['angle'])

    return octahedron;
}

function cube(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    var L = params['length']
    const geometry = new THREE.BoxBufferGeometry(L, L, L);
    const material = new THREE.MeshToonMaterial({transparent: true});
    const cube = new THREE.Mesh(geometry, material);

    var geo = new THREE.EdgesGeometry( cube.geometry );
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2,
                                           opacity: 1, transparent: true });
    var wireframe = new THREE.LineSegments(geo, mat);
    wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd

    cube.add(wireframe);

    cube.position.x = x;
    cube.position.y = y;
    cube.position.z = z;

    return cube;
}

function triangle(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.BufferGeometry();

    var vertices = new Float32Array([
        x + params['vertices'][0][0], y + params['vertices'][0][1], z + params['vertices'][0][2],
        x + params['vertices'][1][0], y + params['vertices'][1][1], z + params['vertices'][1][2],
        x + params['vertices'][2][0], y + params['vertices'][2][1], z + params['vertices'][2][2],
    ]);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    const material = new THREE.MeshBasicMaterial({transparent: true,
                                                  side: THREE.DoubleSide});
    const triangle = new THREE.Mesh(geometry, material);

    var geo = new THREE.EdgesGeometry(triangle.geometry);
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 1,
                                           transparent: true});

    var wireframe = new THREE.LineSegments(geo, mat);
    wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
    triangle.add(wireframe);

    // triangle.position.x = x;
    // triangle.position.y = y;
    // triangle.position.z = z;

    return triangle;
}