import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode, stringToArray } from './base.js';

export {XCubeCode};

class XCubeCode extends AbstractCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, scene);

        this.Lx = size[0];
        this.Ly = size[1];
        this.Lz = size[2];

        this.qubitIndex = qubitIndex;
        this.faceIndex = stabilizerIndex['face'];
        this.cubeIndex = stabilizerIndex['vertex'];

        this.stabilizers = new Array(Hx.length);

        this.toggleStabFn['face'] = this.toggleFace;
        this.toggleStabFn['cube'] = this.toggleCube;

        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};
        this.COLOR = Object.assign(this.COLOR, {
            activatedFace: 0xfa7921,
            activatedCube: 0xf1c232,
            deactivatedCube: 0xf2f28c,
            deactivatedFace: 0xf2f2cc,
            deactivatedEdge: 0xffbcbc,
        });

        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.6,

            activatedStabilizer: {'face': 0.8, 'cube': 0.6},
            minDeactivatedStabilizer: {'face': 0., 'cube': 0},
            maxDeactivatedStabilizer: {'face': 0., 'cube': 0}
        }
    }

    getIndexQubit(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;
        return this.qubitIndex[key];
    }

    getIndexFace(axis, x, y, z) {
        let key = `[${axis}, ${x}, ${y}, ${z}]`;
        return this.faceIndex[key];
    }

    getIndexCube(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;
        return this.cubeIndex[key];
    }

    toggleCube(cube, activate) {
        cube.isActivated = activate;
        let color = activate ? this.COLOR.activatedCube : this.COLOR.deactivatedCube;
        cube.material.color.setHex(color);

        if (activate) {
            cube.material.opacity = this.OPACITY.activatedStabilizer['cube'];
            cube.children[0].material.opacity = this.OPACITY.activatedStabilizer['cube'];
        }
        else {
            cube.material.opacity = this.opacityActivated ? 
                this.OPACITY.minDeactivatedStabilizer['cube'] : this.OPACITY.maxDeactivatedStabilizer['cube'];
            cube.children[0].material.opacity = this.opacityActivated ? 
                this.OPACITY.minDeactivatedStabilizer['cube'] : this.OPACITY.maxDeactivatedStabilizer['cube'];
        }
    }

    toggleFace(face, activate) {
        face.isActivated = activate;
        let color = activate ? this.COLOR.activatedFace : this.COLOR.deactivatedFace;
        face.material.color.setHex(color);

        if (activate) {
            face.material.opacity = this.OPACITY.activatedStabilizer['face'];
            face.visible = true;
        }
        else {
            face.material.opacity = this.opacityActivated ?
                this.OPACITY.minDeactivatedStabilizer['face'] : this.OPACITY.maxDeactivatedStabilizer['face'];
            face.visible = false
        }
    }

    buildQubit(x, y, z) {
        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);

        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedEdge, 
                                                      opacity: this.OPACITY.maxDeactivatedQubit, 
                                                      transparent: true});
        const edge = new THREE.Mesh(geometry, material);

        edge.position.x = x * this.SIZE.lengthEdge / 2;
        edge.position.y = y * this.SIZE.lengthEdge / 2;
        edge.position.z = z * this.SIZE.lengthEdge / 2;

        let x_axis = ((z % 2 == 0) && y % 2 == 0);
        let y_axis = ((z % 2 == 0) && x % 2 == 0);
        let z_axis = (z % 2 == 1);

        if (x_axis) {
            edge.rotateZ(Math.PI / 2)
        }
        else if (z_axis) {
            edge.rotateX(Math.PI / 2)
        }

        edge.hasError = {'X': false, 'Z': false};

        let index = this.getIndexQubit(x, y, z)

        edge.index = index;
        this.qubits[index] = edge;

        this.scene.add(edge);
    }

    buildFace(axis, x, y, z) {
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedFace, 
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
    
        let index = this.getIndexFace(axis, x, y, z);
    
        face.index = index;
        face.type = 'face';
        face.isActivated = false;
    
        this.stabilizers[index] = face;
    
        this.scene.add(face);
    }

    buildCube(x, y, z) {
        const L = this.SIZE.lengthEdge - 0.3
        const geometry = new THREE.BoxBufferGeometry(L, L, L);
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedCube, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['cube'], 
                                                     transparent: true});
        const cube = new THREE.Mesh(geometry, material);

        cube.position.x = x * this.SIZE.lengthEdge / 2;
        cube.position.y = y * this.SIZE.lengthEdge / 2;
        cube.position.z = z * this.SIZE.lengthEdge / 2;

        let index = this.getIndexCube(x, y, z);
        cube.index = index;
        cube.type = 'cube';
        cube.isActivated = false;
        this.stabilizers[index] = cube;

        var geo = new THREE.EdgesGeometry( cube.geometry );
        var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2, 
                                               opacity: this.OPACITY.maxDeactivatedStabilizer['cube'], 
                                               transparent: true });

        var wireframe = new THREE.LineSegments(geo, mat);
        wireframe.renderOrder = 0; // make sure wireframes are rendered 2nd
        cube.add(wireframe);

        this.scene.add(cube);
    }


    build() {
        for (const [coord, index] of Object.entries(this.qubitIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildQubit(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.cubeIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildCube(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.faceIndex)) {
            let [axis, x, y, z] = stringToArray(coord)
            this.buildFace(axis, x, y, z)
        }
    }
}
