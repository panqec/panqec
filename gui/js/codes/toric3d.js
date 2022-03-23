import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';

import { AbstractCubicCode, AbstractRpCubicCode} from './base/abstractCubicCode.js';

export {Toric3DCode, RpToric3DCode};

class Toric3DCode extends AbstractCubicCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene);
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

        let key = `[${x}, ${y}, ${z}]`;
    
        if (this.qubitAxis[key] == 0) {
            edge.rotateZ(Math.PI / 2)
        }
        else if (this.qubitAxis[key] == 2) {
            edge.rotateX(Math.PI / 2)
        }
    
        edge.hasError = {'X': false, 'Z': false};
    
        let index = this.getIndexQubit(x, y, z)
    
        edge.index = index;
        edge.location = [x, y, z];
        this.qubits[index] = edge;
    
        this.scene.add(edge);
    }

    buildVertex(x, y, z) {
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedVertex, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['vertex'], 
                                                     transparent: true});
        const sphere = new THREE.Mesh(geometry, material);
    
        sphere.position.x = x * this.SIZE.lengthEdge / 2;
        sphere.position.y = y * this.SIZE.lengthEdge / 2;
        sphere.position.z = z * this.SIZE.lengthEdge / 2;
    
        let index = this.getIndexVertex(x, y, z);
    
        sphere.index = index;
        sphere.location = [x, y, z];
        sphere.type = 'vertex';
        sphere.isActivated = false;
    
        this.stabilizers[index] = sphere;
    
        this.scene.add(sphere);
    }

    buildFace(x, y, z) {
        const geometry = new THREE.PlaneGeometry(this.SIZE.lengthEdge-0.3, this.SIZE.lengthEdge-0.3);
    
        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'], 
                                                     transparent: true, 
                                                     side: THREE.DoubleSide});
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
        face.location = [x, y, z];
        face.type = 'face';
        face.isActivated = false;
    
        this.stabilizers[index] = face;
    
        this.scene.add(face);
    }
}

class RpToric3DCode extends AbstractRpCubicCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene);
    }

    buildQubit(x, y, z) {
        let length = this.SIZE.lengthEdge;
        const geometry = new THREE.SphereGeometry(this.SIZE.radiusVertex, 32, 32);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.deactivatedQubit, 
                                                     opacity: this.OPACITY.maxDeactivatedQubit,
                                                     transparent: true});
        const sphere = new THREE.Mesh(geometry, material);

        sphere.position.x = x * length / 2;
        sphere.position.y = y * length / 2;
        sphere.position.z = z * length / 2;

        sphere.hasError = {'X': false, 'Z': false};
        let index = this.getIndexQubit(x, y, z);

        sphere.index = index;
        sphere.location = [x, y, z];

        this.qubits[index] = sphere;

        this.scene.add(sphere);
    }

    buildOctahedron(x, y, z) {
        let length = this.SIZE.lengthEdge
        const geometry = new THREE.OctahedronGeometry(this.SIZE.lengthEdge/2);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedOctahedron, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['octahedron'], 
                                                     transparent: true, side: THREE.DoubleSide});
        const octa = new THREE.Mesh(geometry, material);

        octa.position.x = x * this.SIZE.lengthEdge / 2;
        octa.position.y = y * this.SIZE.lengthEdge / 2;
        octa.position.z = z * this.SIZE.lengthEdge / 2;

        if (z % 2 == 0) {
            octa.rotateX(Math.PI/2)

            if ((x + y) % 4 == 2) {
                octa.rotateY(Math.PI/2)
            }
        }

        let index = this.getIndexOctahedron(x, y, z);

        octa.index = index;
        octa.location = [x, y, z];
        octa.type = 'octahedron';
        octa.isActivated = false;

        this.stabilizers[index] = octa;
        this.scene.add(octa);
    }

    buildFace(x, y, z) {
        let length = this.SIZE.lengthEdge
        const geometry = new THREE.PlaneGeometry(length-0.3, length-0.3);

        const material = new THREE.MeshToonMaterial({color: this.COLOR.activatedFace, 
                                                     opacity: this.OPACITY.maxDeactivatedStabilizer['face'], 
                                                     transparent: true, side: THREE.DoubleSide});
        const face = new THREE.Mesh(geometry, material);

        face.position.x = x * length / 2;
        face.position.y = y * length / 2;
        face.position.z = z * length / 2;

        // Axis normal to the face
        let x_axis = ((z % 2 == 1) && (x % 2 == 0));
        let y_axis = ((z % 2 == 1) && (y % 2 == 0));
        let z_axis = (z % 2 == 0);
    
        if (x_axis) {
            face.rotateY(Math.PI / 2)
            face.rotateZ(Math.PI / 4)
        }
        else if (y_axis) {
            face.rotateX(Math.PI / 2)
            face.rotateZ(Math.PI / 4)
        }
        else if (z_axis) {
            face.rotateZ(Math.PI / 4)
        }

        let index = this.getIndexFace(x, y, z);

        face.index = index;
        face.location = [x, y, z];
        face.type = 'face';
        face.isActivated = false;

        this.stabilizers[index] = face;

        this.scene.add(face);
    }
}
