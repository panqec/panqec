import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode } from './base/abstractCode.js';

export {RhombicCode};

class RhombicCode extends AbstractCode {
    COLOR = {
        activatedStabilizer: {'triangle': 0xf1c232, 'cube': 0xf1c232},
        deactivatedStabilizer: {'triangle': 0xf2f2cc, 'cube': 0xf2f2cc},

        deactivatedQubit: 0xffbcbc,
        errorX: 0xFF4B3E,
        errorZ: 0x4381C1,
        errorY: 0x058C42
    };

    OPACITY = {
        minActivatedQubit: 1,
        maxActivatedQubit: 1,
        minDeactivatedQubit: 0.1,
        maxDeactivatedQubit: 0.6,

        minActivatedStabilizer: {'triangle': 1, 'cube': 0.6},
        maxActivatedStabilizer: {'triangle': 1, 'cube': 0.6},
        minDeactivatedStabilizer: {'triangle': 0.1, 'cube': 0.1},
        maxDeactivatedStabilizer: {'triangle': 0.6, 'cube': 0.3}
    };

    constructor(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType) {
        super(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType);
    }

    buildQubit(index) {
        let [x, y, z] = this.qubitCoordinates[index];

        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);

        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedQubit,
                                                      opacity: this.OPACITY.maxDeactivatedQubit,
                                                      transparent: true});
        const qubit = new THREE.Mesh(geometry, material);

        qubit.position.x = x * this.SIZE.lengthEdge / 2;
        qubit.position.y = y * this.SIZE.lengthEdge / 2;
        qubit.position.z = z * this.SIZE.lengthEdge / 2;

        if (this.qubitAxis[index] == 0) {
            qubit.rotateZ(Math.PI / 2);
        }
        else if (this.qubitAxis[index] == 2) {
            qubit.rotateX(Math.PI / 2);
        }

        return qubit;
    }

    buildTriangle(index) {
        let [axis, x, y, z] = this.stabilizerCoordinates[index];

        const L = this.SIZE.lengthEdge / 4

        const geometry = new THREE.BufferGeometry();

        let pos_x = x * this.SIZE.lengthEdge / 2;
        let pos_y = y * this.SIZE.lengthEdge / 2;
        let pos_z = z * this.SIZE.lengthEdge / 2;

        let delta_1 = [[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]];
        let delta_2 = [[1, 1, -1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1]];

        let delta = ((x + y + z) % 4 == 0) ? delta_1 : delta_2

        var vertices = new Float32Array([
            pos_x + delta[axis][0]*L, pos_y,                    pos_z,
            pos_x,                    pos_y + delta[axis][1]*L, pos_z,
            pos_x,                    pos_y,                    pos_z+delta[axis][2]*L
        ]);

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3) );

        const material = new THREE.MeshBasicMaterial({color: this.COLOR.deactivatedStabilizer['triangle'],
                                                      opacity: this.OPACITY.maxDeactivatedStabilizer['triangle'],
                                                      transparent: true,
                                                      side: THREE.DoubleSide});
        const triangle = new THREE.Mesh(geometry, material);

        var geo = new THREE.EdgesGeometry(triangle.geometry);
        var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 1,
                                               opacity: this.OPACITY.maxDeactivatedStabilizer['triangle'],
                                               transparent: true});
        var wireframe = new THREE.LineSegments(geo, mat);
        wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
        triangle.add(wireframe);

        return triangle;
    }

    buildCube(index) {
        let [x, y, z] = this.stabilizerCoordinates[index];

        const L = this.SIZE.lengthEdge - 0.3
        const geometry = new THREE.BoxBufferGeometry(L, L, L);
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedStabilizer['cube'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['cube'],
                                                     transparent: true});
        const cube = new THREE.Mesh(geometry, material);

        var geo = new THREE.EdgesGeometry( cube.geometry );
        var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2,
                                               opacity: this.OPACITY.maxDeactivatedStabilizer['cube'],
                                               transparent: true });
        var wireframe = new THREE.LineSegments( geo, mat );
        wireframe.renderOrder = 1; // make sure wireframes are rendered 2nd
        cube.add(wireframe);

        cube.position.x = x * this.SIZE.lengthEdge / 2;
        cube.position.y = y * this.SIZE.lengthEdge / 2;
        cube.position.z = z * this.SIZE.lengthEdge / 2;

        return cube
    }

    buildStabilizer(index) {
        if (this.stabilizerType[index] == 'cube') {
            var stabilizer = this.buildCube(index);
        }
        else {
            var stabilizer = this.buildTriangle(index);
        }

        return stabilizer;
    }

}
