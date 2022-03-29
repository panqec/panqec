import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode } from './base/abstractCode.js';

export {XCubeCode};

class XCubeCode extends AbstractCode {
    COLOR = {
        activatedStabilizer: {'face': 0xfa7921, 'cube': 0xf1c232},
        deactivatedStabilizer: {'face': 0xf2f2cc, 'cube': 0xf2f28c},

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

        minActivatedStabilizer: {'face': 0.8, 'cube': 0.6},
        maxActivatedStabilizer: {'face': 0.8, 'cube': 0.6},
        minDeactivatedStabilizer: {'face': 0, 'cube': 0},
        maxDeactivatedStabilizer: {'face': 0, 'cube': 0}
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
            qubit.rotateZ(Math.PI / 2)
        }
        else if (this.qubitAxis[index] == 2) {
            qubit.rotateX(Math.PI / 2)
        }

        return qubit
    }

    buildFace(index) {
        let [axis, x, y, z] = this.stabilizerCoordinates[index];

        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedStabilizer['face'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'],
                                                     transparent: true,
                                                     side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);
        face.visible = false;

        face.position.x = x * this.SIZE.lengthEdge / 2;
        face.position.y = y * this.SIZE.lengthEdge / 2;
        face.position.z = z * this.SIZE.lengthEdge / 2;

        if (axis == 0) {
            face.rotateY(Math.PI / 2)
        }
        else if (axis == 1) {
            face.rotateX(Math.PI / 2)
        }
        else {
            face.rotateZ(Math.PI / 2)
        }

        return face;
    }

    buildCube(index) {
        let [x, y, z] = this.stabilizerCoordinates[index];

        const L = this.SIZE.lengthEdge - 0.3
        const geometry = new THREE.BoxBufferGeometry(L, L, L);
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedStabilizer['cube'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['cube'],
                                                     transparent: true});
        const cube = new THREE.Mesh(geometry, material);

        cube.position.x = x * this.SIZE.lengthEdge / 2;
        cube.position.y = y * this.SIZE.lengthEdge / 2;
        cube.position.z = z * this.SIZE.lengthEdge / 2;

        var geo = new THREE.EdgesGeometry( cube.geometry );
        var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2,
                                               opacity: this.OPACITY.maxDeactivatedStabilizer['cube'],
                                               transparent: true });

        var wireframe = new THREE.LineSegments(geo, mat);
        wireframe.renderOrder = 0; // make sure wireframes are rendered 2nd
        cube.add(wireframe);

        return cube;
    }

    buildStabilizer(index) {
        if (this.stabilizerType[index] == 'cube') {
            var stabilizer = this.buildCube(index);
        }
        else {
            var stabilizer = this.buildFace(index);
        }

        return stabilizer;
    }

    updateStabilizers() {
        let syndrome = this.getSyndrome()

        for (let iStab=0; iStab < this.m; iStab++) {
            this.toggleStabilizer(this.stabilizers[iStab], syndrome[iStab], true);
        }
    }
}
