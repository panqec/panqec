import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode} from './base/abstractCode.js';

export {RotatedToric3DCode, RpRotatedToric3DCode};

class RotatedToric3DCode extends AbstractCode {
    constructor(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType) {
        super(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType);

        this.COLOR = {
            deactivatedStabilizer: {'vertex': 0xf2f2fc, 'face': 0xf1c232},
            activatedStabilizer: {'vertex': 0xf1c232, 'face': 0xf1c232,},
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

            minActivatedStabilizer: {'vertex': 1, 'face': 0.6},
            maxActivatedStabilizer: {'vertex': 1, 'face': 0.6},
            minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0},
            maxDeactivatedStabilizer: {'vertex': 0.6, 'face': 0}
        };
    }

    buildQubit(index) {
        let [x, y, z] = this.qubitCoordinates[index];

        let length = this.SIZE.lengthEdge;
        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);

        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedQubit,
                                                      opacity: this.OPACITY.maxDeactivatedQubit,
                                                      transparent: true});
        const qubit = new THREE.Mesh(geometry, material);

        qubit.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        qubit.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        qubit.position.z = length * z / 2 - this.offset.z;

        if (this.qubitAxis[index] == 2) {
            qubit.rotateX(Math.PI / 2);
        }
        else if (this.qubitAxis[index] == 0) {
            qubit.rotateZ(Math.PI / 4);
        }
        else {
            qubit.rotateZ(-Math.PI / 4);
        }

        return qubit;
    }

    buildVertex(index) {
        let [x, y, z] = this.stabilizerCoordinates[index];

        let length = this.SIZE.lengthEdge;
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedStabilizer['vertex'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['vertex'],
                                                     transparent: true});
        const vertex = new THREE.Mesh(geometry, material);

        vertex.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        vertex.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        vertex.position.z = length * z / 2 - this.offset.z;

        return vertex;
    }

    buildFace(index) {
        let [x, y, z] = this.stabilizerCoordinates[index];

        let length = this.SIZE.lengthEdge
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedStabilizer['face'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'],
                                                     transparent: true,
                                                     side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);

        face.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        face.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        face.position.z = length * z / 2 - this.offset.z;

        face.rotateZ(Math.PI/4)

        if (z % 2 == 0) {
            face.rotateX(Math.PI/2)

            if ((x + y) % 4 == 0) {
                face.rotateY(Math.PI/2)
            }
        }

        return face;
    }

    buildStabilizer(index) {
        if (this.stabilizerType[index] == 'vertex') {
            var stabilizer = this.buildVertex(index);
        }
        else {
            var stabilizer = this.buildFace(index);
        }

        return stabilizer;
    }
}

class RpRotatedToric3DCode extends AbstractCode {
    constructor(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType) {
        super(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis,  stabilizerType);

        this.COLOR = {
            deactivatedStabilizer: {'vertex': 0xfa7921, 'face': 0xf1c232},
            activatedStabilizer: {'vertex': 0xfa7921, 'face': 0xf1c232},
            deactivatedQubit: 0xffbcbc,
            errorX: 0xFF4B3E,
            errorZ: 0x4381C1,
            errorY: 0x058C42
        };

        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.4,

            minActivatedStabilizer: {'vertex': 0.9, 'face': 0.9},
            maxActivatedStabilizer: {'vertex': 0.9, 'face': 0.9},
            minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0.1},
            maxDeactivatedStabilizer: {'vertex': 0.3, 'face': 0.3}
        };
    }

    buildQubit(index) {
        let [x, y, z] = this.qubitCoordinates[index];

        let length = this.SIZE.lengthEdge;
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedQubit,
                                                     opacity: this.OPACITY.maxDeactivatedQubit,
                                                     transparent: true});
        const qubit = new THREE.Mesh(geometry, material);

        qubit.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        qubit.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        qubit.position.z = length * z / 2 - this.offset.z;

        return qubit;
    }

    buildOctahedron(index) {
        let [x, y, z] = this.stabilizerCoordinates[index];

        let length = this.SIZE.lengthEdge
        const geometry = new THREE.OctahedronGeometry(this.SIZE.lengthEdge/2);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedStabilizer['vertex'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['vertex'],
                                                     transparent: true, side: THREE.DoubleSide});
        const octa = new THREE.Mesh(geometry, material);

        octa.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        octa.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        octa.position.z = length * z / 2 - this.offset.z;

        octa.rotateZ(Math.PI/4)

        if (z % 2 == 0) {
            octa.rotateX(Math.PI/2);

            if ((x + y) % 4 == 0) {
                octa.rotateY(Math.PI/2);
            }
        }

        return octa;
    }

    buildFace(index) {
        let [x, y, z] = this.stabilizerCoordinates[index];

        let length = this.SIZE.lengthEdge
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedStabilizer['face'],
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'],
                                                     transparent: true, side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);

        face.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        face.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        face.position.z = length * z / 2 - this.offset.z;

        if (z % 2 == 0) {
            face.rotateX(Math.PI/2);

            if ((x + y) % 4 != 0) {
                face.rotateY(Math.PI/4);
            }
            else {
                face.rotateY(-Math.PI/4);
            }

            face.rotateZ(Math.PI/4);
        }

        return face;
    }

    buildStabilizer(index) {
        if (this.stabilizerType[index] == 'vertex') {
            var stabilizer = this.buildOctahedron(index);
        }
        else {
            var stabilizer = this.buildFace(index);
        }

        return stabilizer;
    }
}
