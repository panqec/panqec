import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCode, stringToArray } from './base.js';

export {ToricCode3D};

class ToricCode3D extends AbstractCode {
    constructor(size, Hx, Hz, indices, scene) {
        super(Hx, Hz, scene);

        this.Lx = size[0];
        this.Ly = size[1];
        this.Lz = size[2];

        this.vertices = [];
        this.faces = [];

        this.qubitIndex = indices['qubit'];
        this.vertexIndex = indices['vertex'];
        this.faceIndex = indices['face'];

        this.stabilizers['X'] = this.vertices;
        this.stabilizers['Z'] = this.faces;

        this.toggleStabFn['X'] = this.toggleVertex;
        this.toggleStabFn['Z'] = this.toggleFace;

        this.X_AXIS = 1;
        this.Y_AXIS = 2;
        this.Z_AXIS = 0;
        
        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};
    }

    getIndexQubit(axis, x, y, z) {
        let key = `[${axis}, ${x}, ${y}, ${z}]`;
        return this.qubitIndex[key];
    }

    getIndexFace(axis, x, y, z) {
        let key = `[${axis}, ${x}, ${y}, ${z}]`;
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

    buildEdge(axis, x, y, z) {
        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);
    
        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedEdge, opacity: this.currentOpacity, transparent: true});
        const edge = new THREE.Mesh(geometry, material);
    
        edge.position.x = x;
        edge.position.y = y;
        edge.position.z = z;
    
        if (axis == this.X_AXIS) {
            edge.position.y += this.SIZE.lengthEdge / 2
        }
        if (axis == this.Y_AXIS) {
            edge.rotateX(Math.PI / 2)
            edge.position.z += this.SIZE.lengthEdge / 2
        }
        else if (axis == this.Z_AXIS) {
            edge.rotateZ(Math.PI / 2)
            edge.position.x += this.SIZE.lengthEdge / 2
        }
    
        edge.hasError = {'X': false, 'Z': false};
    
        let index = this.getIndexQubit(axis, x, y, z)
    
        edge.index = index;
        this.qubits[index] = edge;
    
        this.scene.add(edge);
    }

    buildVertex(x, y, z) {
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedVertex, opacity: this.currentOpacity, transparent: true});
        const sphere = new THREE.Mesh(geometry, material);
    
        sphere.position.x = x;
        sphere.position.y = y;
        sphere.position.z = z;
    
        let index = this.getIndexVertex(x, y, z);
    
        sphere.index = index;
        sphere.isActivated = false;
    
        this.vertices[index] = sphere;
    
        this.scene.add(sphere);
    }

    buildFace(axis, x, y, z) {
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, opacity: 0, transparent: true, side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);
    
        face.position.x = x;
        face.position.y = y;
        face.position.z = z;
    
        if (axis == this.Y_AXIS) {
            face.position.x += this.SIZE.lengthEdge / 2;
            face.position.y += this.SIZE.lengthEdge / 2;
        }
        if (axis == this.X_AXIS) {
            face.position.x += this.SIZE.lengthEdge / 2;
            face.position.z += this.SIZE.lengthEdge / 2;
            face.rotateX(Math.PI / 2)
        }
        else if (axis == this.Z_AXIS) {
            face.position.z += this.SIZE.lengthEdge / 2
            face.position.y += this.SIZE.lengthEdge / 2
            face.rotateY(Math.PI / 2)
        }
    
        let index = this.getIndexFace(axis, x, y, z);
    
        face.index = index;
        face.isActivated = false;
    
        this.faces[index] = face;
    
        this.scene.add(face);
    }

    build() {
        for (const [coord, index] of Object.entries(this.qubitIndex)) {
            let [axis, x, y, z] = stringToArray(coord)
            this.buildEdge(axis, x, y, z)
        }
        for (const [coord, index] of Object.entries(this.vertexIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildVertex(x, y, z)
        }
        for (const [coord, index] of Object.entries(this.faceIndex)) {
            let [axis, x, y, z] = stringToArray(coord)
            this.buildFace(axis, x, y, z)
        }
    }
}
