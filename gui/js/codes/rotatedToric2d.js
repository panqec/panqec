import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractSurfaceCode, AbstractRpSurfaceCode} from './base/abstractSurfaceCode.js';

export {RotatedToric2DCode, RpRotatedToric2DCode};

class RotatedToric2DCode extends AbstractSurfaceCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, scene);

        this.SIZE = {radiusEdge: 0.05 / (this.Lx/4), radiusVertex: 0.1 / (this.Lx/4), lengthEdge: 0.5 / (0.05 + this.Lx/4)};

        this.offset = {
            x: this.SIZE.lengthEdge * (this.Lx) / 2 - this.SIZE.lengthEdge/4,
            y: this.SIZE.lengthEdge * (this.Ly) / 2 - this.SIZE.lengthEdge/4
        };
    }

    buildQubit(x, y) {
        const geometry = new THREE.CylinderGeometry(this.SIZE.radiusEdge, this.SIZE.radiusEdge, this.SIZE.lengthEdge, 32);
    
        const material = new THREE.MeshPhongMaterial({color: this.COLOR.deactivatedQubit, 
                                                      opacity: this.OPACITY.maxDeactivatedQubit, 
                                                      transparent: true});
        const edge = new THREE.Mesh(geometry, material);
    
        edge.position.x = x * this.SIZE.lengthEdge / 2 - this.offset.x;
        edge.position.y = y * this.SIZE.lengthEdge / 2 - this.offset.y;

        let x_axis = ((x + y) % 4 == 2);
    
        if (x_axis) {
            edge.rotateZ(Math.PI / 4)
        }
        else {
            edge.rotateZ(-Math.PI / 4)
        }
    
        edge.hasError = {'X': false, 'Z': false};
    
        let index = this.getIndexQubit(x, y)
    
        edge.index = index;
        this.qubits[index] = edge;
    
        this.scene.add(edge);
    }

    buildVertex(x, y) {
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedVertex, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['vertex'], 
                                                     transparent: true});
        const sphere = new THREE.Mesh(geometry, material);
    
        sphere.position.x = x * this.SIZE.lengthEdge / 2  - this.offset.x;
        sphere.position.y = y * this.SIZE.lengthEdge / 2  - this.offset.y;
    
        let index = this.getIndexVertex(x, y);
    
        sphere.index = index;
        sphere.type = 'vertex';
        sphere.isActivated = false;
    
        this.stabilizers[index] = sphere;
    
        this.scene.add(sphere);
    }

    buildFace(x, y) {
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge*Math.sqrt(2), this.SIZE.lengthEdge*Math.sqrt(2));
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'], 
                                                     transparent: true, 
                                                     side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);
    
        face.position.x = x * this.SIZE.lengthEdge / 2 - this.offset.x;
        face.position.y = y * this.SIZE.lengthEdge / 2 - this.offset.y;

        face.rotateZ(Math.PI / 4)

        let index = this.getIndexFace(x, y);
    
        face.index = index;
        face.type = 'face';
        face.isActivated = false;
    
        this.stabilizers[index] = face;
    
        this.scene.add(face);
    }
}


class RpRotatedToric2DCode extends AbstractRpSurfaceCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, scene);
    }

    buildQubit(x, y) {
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedQubit, 
                                                     opacity: this.OPACITY.maxDeactivatedQubit, 
                                                     transparent: true});
        const qubit = new THREE.Mesh(geometry, material);
    
        qubit.position.x = x * this.SIZE.lengthEdge / 2  - this.offset.x;
        qubit.position.y = y * this.SIZE.lengthEdge / 2  - this.offset.y;
    
        let index = this.getIndexQubit(x, y);
    
        qubit.index = index;
        qubit.hasError = {'X': false, 'Z': false};
        this.qubits[index] = qubit;
        
        this.scene.add(qubit);
    }

    buildVertex(x, y) {
        // In the rotated picture, vertices are represented by faces with a different colors

        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge, this.SIZE.lengthEdge);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedVertex, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['vertex'], 
                                                     transparent: true});
        const vertex = new THREE.Mesh(geometry, material);
    
        vertex.position.x = x * this.SIZE.lengthEdge / 2 - this.offset.x;
        vertex.position.y = y * this.SIZE.lengthEdge / 2 - this.offset.y;

        // vertex.rotateZ(Math.PI/4)

        let index = this.getIndexVertex(x, y);
    
        vertex.index = index;
        vertex.type = 'vertex';
        vertex.isActivated = false;
    
        this.stabilizers[index] = vertex;
    
        this.scene.add(vertex);
    }

    buildFace(x, y) {
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge, this.SIZE.lengthEdge);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedFace, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'], 
                                                     transparent: true, 
                                                     side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);
    
        face.position.x = x * this.SIZE.lengthEdge / 2 - this.offset.x;
        face.position.y = y * this.SIZE.lengthEdge / 2 - this.offset.y;

        // face.rotateZ(Math.PI/4)

        let index = this.getIndexFace(x, y);
    
        face.index = index;
        face.type = 'face';
        face.isActivated = false;
    
        this.stabilizers[index] = face;
    
        this.scene.add(face);
    }
}
