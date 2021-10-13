import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode, stringToArray } from './base.js';

export {RotatedCode3D};

class RotatedCode3D extends AbstractCode {
    constructor(Lx, Ly, Lz, Hx, Hz, indices, scene) {
        super(Hx, Hz, scene);

        this.Lx = Lx;
        this.Ly = Ly;
        this.Lz = Lz;

        this.vertices = new Array(Hx.length);
        this.faces = new Array(Hz.length);

        this.qubitIndex = indices['qubit'];
        this.vertexIndex = indices['vertex'];
        this.faceIndex = indices['face'];

        this.stabilizers['X'] = this.vertices;
        this.stabilizers['Z'] = this.faces;

        this.toggleStabFn['X'] = this.toggleVertex;
        this.toggleStabFn['Z'] = this.toggleFace;
        
        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};
        this.COLOR = {deactivatedVertex: 0xf2f28c, activatedVertex: 0xf1c232,
                      deactivatedEdge: 0xffbcbc, activatedFace: 0xf1c232, 
                      errorX: 0xff0000, errorZ: 0x25CCF7, errorY: 0xa55eea};

        let length = this.SIZE.lengthEdge;
        this.offset = {x: Math.SQRT2 * length*this.L / 2, y: Math.SQRT2 * length*this.L / 2, z: length*this.L / 2};
    }

    getIndexQubit(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;
        return this.qubitIndex[key];
    }

    getIndexFace(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;
        return this.faceIndex[key];
    }
 
    getIndexVertex(x, y, z) {
        let key = `[${x}, ${y}, ${z}]`;

        return this.vertexIndex[key];
    }

    toggleVertex(vertex, activate) {
        vertex.isActivated = activate;
        vertex.material.transparent = !activate;
        let color = activate ? this.COLOR.activatedVertex : this.COLOR.deactivatedVertex;
        vertex.material.color.setHex(color);
    }
    
    toggleFace(face, activate) {
        face.isActivated = activate;
        face.material.opacity = activate ? this.MAX_OPACITY : 0;
    }

    changeOpacity() {
        if (this.currentOpacity == this.MIN_OPACITY) {
            this.currentOpacity = this.MAX_OPACITY;
        }
        else {
            this.currentOpacity = this.MIN_OPACITY;
        }

        this.qubits.forEach(q => {
            if (!q.hasError['X'] && !q.hasError['Z']) {
                q.material.opacity = this.currentOpacity;
            }
        });
    
        this.vertices.forEach(v => {
            if (!v.isActivated) {
                v.material.opacity = this.currentOpacity;
            }
        });
    }

    buildEdge(x, y, z) {
        let length = this.SIZE.lengthEdge
        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);
    
        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedEdge, opacity: this.currentOpacity, transparent: true});
        const edge = new THREE.Mesh(geometry, material);
    
        edge.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        edge.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        edge.position.z = length * z / 2 - this.offset.z;
    

        if (z % 2 == 0) {
            edge.rotateX(Math.PI / 2);
        }
        else if ((x + y) % 4 == 2) {
            edge.rotateZ(Math.PI / 4);
        }
        else if ((x + y) % 4 == 0) {
            edge.rotateZ(-Math.PI / 4);
        }
        else {
            console.error("Coordinate (",x, y, z, ") is not correct")
        }

        // edge.position.z += this.SIZE.lengthEdge

        
    
        edge.hasError = {'X': false, 'Z': false};
        let index = this.getIndexQubit(x, y, z)
        edge.index = index;

        this.qubits[index] = edge;

        this.scene.add(edge);
    }

    buildVertex(x, y, z) {
        let length = this.SIZE.lengthEdge;
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedVertex, opacity: this.currentOpacity, transparent: true});
        const sphere = new THREE.Mesh(geometry, material);
    
        sphere.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        sphere.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        sphere.position.z = length * z / 2 - this.offset.z;
    
        let index = this.getIndexVertex(x, y, z);
    
        sphere.index = index;
        sphere.isActivated = false;
    
        this.vertices[index] = sphere;
    
        this.scene.add(sphere);
    }

    buildFace(x, y, z) {
        let length = this.SIZE.lengthEdge
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, opacity: 0, transparent: true, side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);
    
        face.position.x = (length * Math.SQRT2 / 4) * x - this.offset.x;
        face.position.y = (length * Math.SQRT2 / 4) * y - this.offset.y;
        face.position.z = length * z / 2 - this.offset.z;

        face.rotateZ(Math.PI/4)

        if (z % 2 == 0) {
            face.rotateX(Math.PI/2)

            if ((x + y) % 4 == 2) {
                face.rotateY(Math.PI/2)
            }
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
            this.buildEdge(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.vertexIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildVertex(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.faceIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildFace(x, y, z)
        }
    }
}