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
        
        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};
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
        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);
    
        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedEdge, opacity: this.currentOpacity, transparent: true});
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

    buildVertex(x, y, z) {
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedVertex, opacity: this.currentOpacity, transparent: true});
        const sphere = new THREE.Mesh(geometry, material);
    
        sphere.position.x = x * this.SIZE.lengthEdge / 2;
        sphere.position.y = y * this.SIZE.lengthEdge / 2;
        sphere.position.z = z * this.SIZE.lengthEdge / 2;
    
        let index = this.getIndexVertex(x, y, z);
    
        sphere.index = index;
        sphere.isActivated = false;
    
        this.vertices[index] = sphere;
    
        this.scene.add(sphere);
    }

    buildFace(x, y, z) {
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, opacity: 0, transparent: true, side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);
    
        face.position.x = x * this.SIZE.lengthEdge / 2;
        face.position.y = y * this.SIZE.lengthEdge / 2;
        face.position.z = z * this.SIZE.lengthEdge / 2;

        // Axis normal to the face
        let x_axis = ((z % 2 == 1) && (x % 2 == 0));
        let y_axis = ((z % 2 == 1) && (y % 2 == 0));
        let z_axis = (z % 2 == 0);
    
        if (x_axis) {
            face.rotateY(Math.PI / 2)
        }
        else if (y_axis) {
            face.rotateX(Math.PI / 2)
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
