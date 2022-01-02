import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode, stringToArray } from './base.js';

export {RpRotatedPlanarCode3D};

class RpRotatedPlanarCode3D extends AbstractCode {
    constructor(size, Hx, Hz, indices, scene) {
        super(Hx, Hz, scene);

        this.Lx = size[0];
        this.Ly = size[1];
        this.Lz = size[2];


        this.octahedrons = new Array(Hx.length);
        this.faces = new Array(Hz.length);

        this.qubitIndex = indices['qubit'];
        this.octahedronIndex = indices['vertex'];
        this.faceIndex = indices['face'];

        this.stabilizers['X'] = this.octahedrons;
        this.stabilizers['Z'] = this.faces;

        this.toggleStabFn['X'] = this.toggleOctahedron;
        this.toggleStabFn['Z'] = this.toggleFace;

        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};

        let length = this.SIZE.lengthEdge;
        this.offset = {x: Math.SQRT2 * length*this.Lx / 2, y: Math.SQRT2 * length*this.Ly / 2, z: length*this.Lz / 2};
    }

    getIndexQubit(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;
        return this.qubitIndex[key];
    }

    getIndexOctahedron(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;
        return this.octahedronIndex[key];
    }

    getIndexFace(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;

        return this.faceIndex[key];
    }

    toggleOctahedron(octa, activate) {
        octa.isActivated = activate;
        if (this.opacityActivated) {
            octa.material.opacity = activate ? this.OPACITY.activatedFace : this.OPACITY.minDeactivatedFace;
        }
        else {
            octa.material.opacity = activate ? this.OPACITY.activatedFace : this.OPACITY.maxDeactivatedFace;
        }
    }

    toggleFace(face, activate) {
        face.isActivated = activate;
        if (this.opacityActivated) {
            face.material.opacity = activate ? this.OPACITY.activatedFace : this.OPACITY.minDeactivatedFace;
        }
        else {
            face.material.opacity = activate ? this.OPACITY.activatedFace : this.OPACITY.maxDeactivatedFace;
        }
    }

    changeOpacity() {
        this.opacityActivated = !this.opacityActivated
        console.log(this.opacityActivated)

        this.qubits.forEach(q => {
            if (!q.hasError['X'] && !q.hasError['Z']) {
                q.material.opacity = this.opacityActivated ? this.OPACITY.minDeactivatedVertex : this.OPACITY.maxDeactivatedVertex;
            }
            else {
                q.material.opacity = this.OPACITY.activatedVertex;
            }
        });

        this.octahedrons.forEach(f => {
            if (!f.isActivated) {
                f.material.opacity = this.opacityActivated ? this.OPACITY.minDeactivatedFace : this.OPACITY.maxDeactivatedFace;
            }
            else {
                f.material.opacity = this.OPACITY.activatedFace;
            }
        });

        this.faces.forEach(f => {
            if (!f.isActivated) {
                f.material.opacity = this.opacityActivated ? this.OPACITY.minDeactivatedFace : this.OPACITY.maxDeactivatedFace;
            }
            else {
                f.material.opacity = this.OPACITY.activatedFace;
            }
        });
    }

    buildQubit(x, y, z) {
        let length = this.SIZE.lengthEdge;
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedQubit, 
                                                     opacity: this.OPACITY.maxDeactivatedVertex,
                                                     transparent: true});
        const sphere = new THREE.Mesh(geometry, material);

        sphere.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        sphere.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        sphere.position.z = length * z / 2 - this.offset.z;

        sphere.hasError = {'X': false, 'Z': false};
        let index = this.getIndexQubit(x, y, z)
        sphere.index = index;

        this.qubits[index] = sphere;

        this.scene.add(sphere);
    }

    buildOctahedron(x, y, z) {
        let length = this.SIZE.lengthEdge
        const geometry = new THREE.OctahedronGeometry(this.SIZE.lengthEdge/2);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedOctahedron, 
                                                     opacity: this.OPACITY.maxDeactivatedFace, 
                                                     transparent: true, side: THREE.DoubleSide});
        const octa = new THREE.Mesh(geometry, material);

        octa.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        octa.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        octa.position.z = length * z / 2 - this.offset.z;

        octa.rotateZ(Math.PI/4)

        if (z % 2 == 0) {
            octa.rotateX(Math.PI/2)

            if ((x + y) % 4 == 2) {
                console.log("test X")
                octa.rotateY(Math.PI/2)
            }
        }

        let index = this.getIndexOctahedron(x, y, z);

        octa.index = index;
        octa.isActivated = false;

        this.octahedrons[index] = octa;

        this.scene.add(octa);
    }

    buildFace(x, y, z) {
        let length = this.SIZE.lengthEdge
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, 
                                                     opacity: this.OPACITY.maxDeactivatedFace, 
                                                     transparent: true, side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);

        face.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        face.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        face.position.z = length * z / 2 - this.offset.z;

        if (z % 2 == 0) {
            face.rotateX(Math.PI/2)

            if ((x + y) % 4 != 2) {
                face.rotateY(Math.PI/4)
            }
            else {
                face.rotateY(-Math.PI/4)
            }
            
            face.rotateZ(Math.PI/4)
        }

        let index = this.getIndexFace(x, y, z);

        face.index = index;
        face.isActivated = false;

        this.faces[index] = face;

        this.scene.add(face);
    }

    build() {
        for (const [coord, index] of Object.entries(this.qubitIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildQubit(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.octahedronIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildOctahedron(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.faceIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildFace(x, y, z)
        }
        console.log(this.qubits)
    }
}
