import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { create_shape } from '../shapes.js';
import { AbstractCode } from './base/abstractCode.js';

export {Toric2DCode, RpToric2DCode};

class Toric2DCode extends AbstractCode {
    COLOR = {
        deactivatedStabilizer: {'vertex': 0xf2f2fc, 'face': 0xf1c232},
        activatedStabilizer: {'vertex': 0xf1c232, 'face': 0xf1c232},

        deactivatedQubit: 0xffbcbc,
        errorX: 0xFF4B3E,
        errorZ: 0x48BEFF,
        errorY: 0x058C42
    };

    OPACITY = {
        minActivatedQubit: 1,
        maxActivatedQubit: 1,
        minDeactivatedQubit: 0.1,
        maxDeactivatedQubit: 0.6,

        minActivatedStabilizer: {'vertex': 1, 'face': 0.6},
        maxActivatedStabilizer: {'vertex': 1, 'face': 0.6},
        minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0},
        maxDeactivatedStabilizer: {'vertex': 0.6, 'face': 0}
    }

    buildQubit(index) {
        var axis = ['x', 'y', 'z'];
        var params = {'length': 2, 'radius': this.SIZE.radiusEdge,
                      'axis': axis[this.qubitAxis[index]],
                      'angle': 0};
        var qubit = create_shape['cylinder'](this.qubitCoordinates[index], params);

        return qubit
    }

    buildVertex(index) {
        var params = {'radius': this.SIZE.radiusVertex};
        var vertex = create_shape['sphere'](this.stabilizerCoordinates[index], params);

        return vertex
    }

    buildFace(index) {
        var params = {'w': 1.5, 'h': 1.5, 'angle': 0, 'plane': 'xy'};
        var face = create_shape['face'](this.stabilizerCoordinates[index], params);

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
    COLOR = {
        activatedStabilizer: {'vertex': 0xfabc2a, 'face': 0xFA824C},
        deactivatedStabilizer: {'vertex': 0xFAFAC6, 'face': 0xe79e90},

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

        minActivatedStabilizer: {'vertex': 0.9, 'face': 0.9},
        maxActivatedStabilizer: {'vertex': 0.9, 'face': 0.9},
        minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0.1},
        maxDeactivatedStabilizer: {'vertex': 0.2, 'face': 0.2}
    }

    buildQubit(index) {
        let [x, y] = this.qubitCoordinates[index];

        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedQubit,
                                                     opacity: this.OPACITY.maxDeactivatedQubit,
                                                     transparent: true});
        const qubit = new THREE.Mesh(geometry, material);

        qubit.position.x = x  / 2;
        qubit.position.y = y  / 2;

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

        face.position.x = x  / 2;
        face.position.y = y  / 2;

        face.rotateZ(Math.PI/4)

        return face
    }
}
