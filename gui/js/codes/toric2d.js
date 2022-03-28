import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode } from './base/abstractCode.js';

export {Toric2DCode, RpToric2DCode};

class Toric2DCode extends AbstractCode {
    constructor(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType) {
        super(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType);

        this.COLOR = {
            deactivatedStabilizer: {'vertex': 0xf2f2fc, 'face': 0xf1c232},
            activatedStabilizer: {'vertex': 0xf1c232, 'face': 0xf1c232},

            deactivatedQubit: 0xffbcbc,
            errorX: 0xFF4B3E,
            errorZ: 0x48BEFF,
            errorY: 0x058C42
        };

        this.OPACITY = {
            minActivatedQubit: 1,
            maxActivatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.6,

            minActivatedStabilizer: {'vertex': 1, 'face': 0.6},
            maxActivatedStabilizer: {'vertex': 1, 'face': 0.6},
            minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0},
            maxDeactivatedStabilizer: {'vertex': 0.6, 'face': 0}
        }
    }

    buildQubit(index) {
        let [x, y] = this.qubitCoordinates[index];  

        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);

        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedQubit,
                                                      opacity: this.OPACITY.maxDeactivatedQubit,
                                                      transparent: true});
        const qubit = new THREE.Mesh(geometry, material);

        qubit.position.x = x * this.SIZE.lengthEdge / 2;
        qubit.position.y = y * this.SIZE.lengthEdge / 2;

        if (this.qubitAxis[index] == 0) {
            qubit.rotateZ(Math.PI / 2)
        }

        return qubit
    }

    buildVertex(index) {
        let [x, y] = this.stabilizerCoordinates[index];

        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedStabilizer['vertex'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['vertex'],
                                                     transparent: true});
        const vertex = new THREE.Mesh(geometry, material);

        vertex.position.x = x * this.SIZE.lengthEdge / 2;
        vertex.position.y = y * this.SIZE.lengthEdge / 2;

        return vertex
    }

    buildFace(index) {
        let [x, y] = this.stabilizerCoordinates[index];

        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge/2*Math.sqrt(2), this.SIZE.lengthEdge/2*Math.sqrt(2));

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedStabilizer['face'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'],
                                                     transparent: true,
                                                     side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);

        face.position.x = x * this.SIZE.lengthEdge / 2;
        face.position.y = y * this.SIZE.lengthEdge / 2;

        return face
    }

    buildStabilizer(index) {
        if (this.stabilizerType[index] == 'vertex') {
            var stabilizer = this.buildVertex(index);
        }
        else {
            var stabilizer = this.buildFace(index);
        }

        return stabilizer
    }
}


class RpToric2DCode extends AbstractCode {
    constructor(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType) {
        super(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType);

        this.COLOR = {
            activatedStabilizer: {'vertex': 0xfabc2a, 'face': 0xFA824C},
            deactivatedStabilizer: {'vertex': 0xFAFAC6, 'face': 0xe79e90},

            deactivatedQubit: 0xffbcbc,
            errorX: 0xFF4B3E,
            errorZ: 0x4381C1,
            errorY: 0x058C42
        };

        this.OPACITY = {
            minActivatedQubit: 1,
            maxActivatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.6,

            minActivatedStabilizer: {'vertex': 0.9, 'face': 0.9},
            maxActivatedStabilizer: {'vertex': 0.9, 'face': 0.9},
            minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0.1},
            maxDeactivatedStabilizer: {'vertex': 0.2, 'face': 0.2}
        }
    }

    buildQubit(index) {
        let [x, y] = this.qubitCoordinates[index];

        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedQubit,
                                                     opacity: this.OPACITY.maxDeactivatedQubit,
                                                     transparent: true});
        const qubit = new THREE.Mesh(geometry, material);

        qubit.position.x = x * this.SIZE.lengthEdge / 2;
        qubit.position.y = y * this.SIZE.lengthEdge / 2;

        return qubit
    }

    buildStabilizer(index) {
        let [x, y] = this.stabilizerCoordinates[index];
        let stabilizerType = this.stabilizerType[index]

        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge/2*Math.sqrt(2), this.SIZE.lengthEdge/2*Math.sqrt(2));

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedStabilizer[stabilizerType],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer[stabilizerType],
                                                     transparent: true,
                                                     side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);

        face.position.x = x * this.SIZE.lengthEdge / 2;
        face.position.y = y * this.SIZE.lengthEdge / 2;

        face.rotateZ(Math.PI/4)

        return face
    }
}
