import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

export {create_shape}

var create_shape = {'sphere': sphere,
                    'rectangle': rectangle,
                    'cylinder': cylinder,
                    'octahedron': octahedron,
                    'box': box,
                    'cube': cube,
                    'triangle': triangle,
                    'hexagon': hexagon,
                    'polygon': polygon,
                    'cuboctahedron': cuboctahedron,
                    'truncated_octahedron': truncated_octahedron}

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

function cuboctahedron(location, params) {
    var x = location[0];
    var y = location[1];
    var z = location[2];

    const vertices = [
        -1,0,-1,    0,-1,-1,   1,0,-1,     0,1,-1,
        -1,-1,0,   1,-1,0,   1,1,0,    -1,1,0,
        -1,0,1,    0,-1,1,   1,0,1,     0,1,1,
    ];

    const faceIndices = [
        0,1,2,    2,3,0,
        8,9,10,   10,11,8,
        0,7,8,    8,4,0,
        4,1,5,    5,9,4,
        5,2,6,    6,10,5,
        7,3,6,    6,11,7,
        0,3,7,    0,1,4,
        1,5,2,    2,3,6,
        7,8,11,   4,8,9,
        5,9,10,   6,10,11
    ];

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(faceIndices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshToonMaterial({transparent: true,
                                                 side: THREE.DoubleSide});
    const cuboctahedron = new THREE.Mesh(geometry, material);

    var geo = new THREE.EdgesGeometry( cuboctahedron.geometry );
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2,
                                           opacity: 0, transparent: true });
    var wireframe = new THREE.LineSegments(geo, mat);
    // wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd

    cuboctahedron.add(wireframe);

    cuboctahedron.position.x = x;
    cuboctahedron.position.y = y;
    cuboctahedron.position.z = z;

    // cuboctahedron.rotateZ(params['angle'])

    return cuboctahedron;
}

function box(location, params) {
    var x = location[0];
    var y = location[1];
    var z = location[2];

    var Lx = params['Lx'];
    var Ly = params['Ly'];
    var Lz = params['Lz'];
    const geometry = new THREE.BoxBufferGeometry(Lx, Ly, Lz);
    const material = new THREE.MeshToonMaterial({transparent: true});
    const box = new THREE.Mesh(geometry, material);

    var geo = new THREE.EdgesGeometry( box.geometry );
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2,
                                           opacity: 0, transparent: true });
    var wireframe = new THREE.LineSegments(geo, mat);
    wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd

    box.add(wireframe);

    box.position.x = x;
    box.position.y = y;
    box.position.z = z;

    return box;
}

function cube(location, params) {
    params['Lx'] = params['length'];
    params['Ly'] = params['length'];
    params['Lz'] = params['length'];

    return box(location, params)
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

    return triangle;
}

function hexagon(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;
    var r = params['radius']

    const geometry = new THREE.BufferGeometry();

    var vertices = new Float32Array([
        0, 0, 0,
        0 + r * Math.cos(Math.PI/3), 0 + r * Math.sin(Math.PI/3), 0,
        0 + r * Math.cos(2*Math.PI/3), 0 + r * Math.sin(2*Math.PI/3), 0,

        0, 0, 0,
        0 + r * Math.cos(2*Math.PI/3), 0 + r * Math.sin(2*Math.PI/3), 0,
        0 + r * Math.cos(3*Math.PI/3), 0 + r * Math.sin(3*Math.PI/3), 0,

        0, 0, 0,
        0 + r * Math.cos(3*Math.PI/3), 0 + r * Math.sin(3*Math.PI/3), 0,
        0 + r * Math.cos(4*Math.PI/3), 0 + r * Math.sin(4*Math.PI/3), 0,

        0, 0, 0,
        0 + r * Math.cos(4*Math.PI/3), 0 + r * Math.sin(4*Math.PI/3), 0,
        0 + r * Math.cos(5*Math.PI/3), 0 + r * Math.sin(5*Math.PI/3), 0,

        0, 0, 0,
        0 + r * Math.cos(5*Math.PI/3), 0 + r * Math.sin(5*Math.PI/3), 0,
        0 + r * Math.cos(6*Math.PI/3), 0 + r * Math.sin(6*Math.PI/3), 0,

        0, 0, 0,
        0 + r * Math.cos(6*Math.PI/3), 0 + r * Math.sin(6*Math.PI/3), 0,
        0 + r * Math.cos(Math.PI/3), 0 + r * Math.sin(Math.PI/3), 0,
    ]);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    const material = new THREE.MeshBasicMaterial({transparent: true,
                                                  side: THREE.DoubleSide});
    const hexagon = new THREE.Mesh(geometry, material);

    var normal = {'x': params['normal'][0], 'y': params['normal'][1], 'z': params['normal'][2]};
    var norm = Math.sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    var norm_xz = Math.sqrt(normal.x*normal.x + normal.z*normal.z);
    var norm_xy = Math.sqrt(normal.x*normal.x + normal.y*normal.y);

    var theta = (norm_xz == 0) ? Math.PI / 2 : Math.acos(normal.z / norm_xz);
    theta = (normal.x >= 0) ? theta : -theta;
    var alpha = (norm_xy == 0) ? 0 : Math.acos(normal.x / norm_xy);
    alpha = (normal.y > 0) ? alpha : -alpha;

    hexagon.geometry.rotateZ(params['angle']);

    hexagon.geometry.rotateY(theta);
    hexagon.geometry.rotateZ(alpha);

    hexagon.geometry.translate(x, y, z);

    var geo = new THREE.EdgesGeometry(hexagon.geometry);
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 1,
                                           transparent: true});

    var wireframe = new THREE.LineSegments(geo, mat);
    wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
    hexagon.add(wireframe);

    return hexagon;
}

function polygon(location, params) {
    var x = location[0];
    var y = location[1];
    var z = (location.length == 3) ? location[2] : 0;

    const geometry = new THREE.BufferGeometry();

    var vertices = params['vertices'];

    // A polygon is defined by many triangles
    var verticesTriangle = [];

    for (var i = 0; i < vertices.length; i++) {
        verticesTriangle.push(
            0, 0, 0,
            vertices[i][0], vertices[i][1], 0,
            vertices[(i + 1) % vertices.length][0], vertices[(i + 1) % vertices.length][1], 0
        )
    }

    var verticesTriangle = new Float32Array(verticesTriangle);

    geometry.setAttribute('position', new THREE.BufferAttribute(verticesTriangle, 3));

    const material = new THREE.MeshBasicMaterial({transparent: true,
                                                  side: THREE.DoubleSide});
    const polygon = new THREE.Mesh(geometry, material);

    var normal = {'x': params['normal'][0], 'y': params['normal'][1], 'z': params['normal'][2]};
    var norm = Math.sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    var norm_xz = Math.sqrt(normal.x*normal.x + normal.z*normal.z);
    var norm_xy = Math.sqrt(normal.x*normal.x + normal.y*normal.y);

    var theta = (norm_xz == 0) ? Math.PI / 2 : Math.acos(normal.z / norm_xz);
    theta = (normal.x >= 0) ? theta : -theta;
    var alpha = (norm_xy == 0) ? 0 : Math.acos(normal.x / norm_xy);
    alpha = (normal.y > 0) ? alpha : -alpha;

    polygon.geometry.rotateZ(params['angle']);

    polygon.geometry.rotateY(theta);
    polygon.geometry.rotateZ(alpha);

    polygon.geometry.translate(x, y, z);

    var geo = new THREE.EdgesGeometry(polygon.geometry);
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 1,
                                           transparent: true});

    var wireframe = new THREE.LineSegments(geo, mat);
    wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
    polygon.add(wireframe);

    return polygon;
}

function faces2triangles(faces) {
    let acc = [];
    faces.forEach(function(f1) {
        for (let i = 1; i < f1.length - 1; i++) {
        acc = acc.concat([f1[0], f1[i], f1[i+1]]) }
    })

    return acc
}

function truncated_octahedron(location, params) {
    // Thanks Nascif (https://observablehq.com/@nascif/truncated-octahedron-take-1)

    var x = location[0];
    var y = location[1];
    var z = location[2];

    const vertices = [
        0,1,2, 0,1,-2,  0,-1,2, 0,-1,-2,
        1,0,2, 1,0,-2, -1,0,2, -1,0,-2,
        1,2,0, 1,-2,0, -1,2,0, -1,-2,0,
        0,2,1, 0,2,-1,  0,-2,1, 0,-2,-1,
        2,0,1, 2,0,-1, -2,0,1, -2,0,-1,
        2,1,0, 2,-1,0, -2,1,0, -2,-1,0
    ]

    const faces = [
        [6,2,14,11,23,18],
        [12,8,20,16,4,0],
        [2,4,16,21,9,14],
        [0,6,18,22,10,12],
        [1,5,17,20,8,13],
        [15,9,21,17,5,3],
        [19,7,1,13,10,22],
        [3,7,19,23,11,15],
        [0,4,2,6],
        [1,7,3,5],
        [8,12,10,13],
        [15,11,14,9],
        [23,19,22,18],
        [21,16,20,17]
    ]

    const triangles = faces2triangles(faces)

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(triangles);
    geometry.computeVertexNormals();

    const material = new THREE.MeshToonMaterial({transparent: true,
                                                 side: THREE.DoubleSide});
    const trunc_octahedron = new THREE.Mesh(geometry, material);

    var geo = new THREE.EdgesGeometry( trunc_octahedron.geometry );
    var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2,
                                           opacity: 0, transparent: true });
    var wireframe = new THREE.LineSegments(geo, mat);
    // wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd

    trunc_octahedron.add(wireframe);

    trunc_octahedron.position.x = x;
    trunc_octahedron.position.y = y;
    trunc_octahedron.position.z = z;

    // trunc_octahedron.rotateZ(params['angle'])

    return trunc_octahedron;

}