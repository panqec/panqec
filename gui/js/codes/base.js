export {AbstractCode, AbstractCubicCode, AbstractRpCubicCode, stringToArray};

function stringToArray(a) {
    return a.replace("[", "").replace("]", "").split(", ").map(Number)
}

class AbstractCode {
    constructor(size, Hx, Hz, indices, scene) {
        this.H = {'X': Hx, 'Z': Hz};
        this.scene = scene;

        this.Lx = size[0];
        this.Ly = size[1];
        if (size.length > 2) {
            this.Lz = size[2];
        }

        this.opacityActivated = false;
        this.currentIndexLogical = {'X': 0, 'Z': 0}
        
        this.stabilizers = {'X': [], 'Z': []};
        this.toggleStabFn = {'X': 0, 'Z': 0};
        this.qubits = new Array(Hx[0].length);
        this.COLOR = {
            deactivatedVertex: 0xf2f28c,
            activatedVertex: 0xf1c232,
            deactivatedEdge: 0xffbcbc,
            deactivatedQubit: 0xffbcbc,
            activatedFace: 0xf1c232, 
            activatedOctahedron: 0xfa7921, 
            errorX: 0xFF4B3E,
            errorZ: 0x4A5899,
            errorY: 0x058C42
        };

        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};
        let length = this.SIZE.lengthEdge;
        this.offset = {x: Math.SQRT2 * length*this.Lx / 2, y: Math.SQRT2 * length*this.Ly / 2, z: length*this.Lz / 2};
    }

    updateStabilizers() {
        let nQubitErrors;
        for (let pauli of ['X' , 'Z']) {
            for (let iStab=0; iStab < this.H[pauli].length; iStab++) {
                nQubitErrors = 0
                for (let iQubit=0; iQubit < this.H[pauli][0].length; iQubit++) {
                    if (this.H[pauli][iStab][iQubit] == 1) {
                        if (this.qubits[iQubit].hasError[pauli]) {
                            nQubitErrors += 1
                        }
                    }
                }
                let activate = (nQubitErrors % 2 == 1);
                this.toggleStabFn[pauli].call(this, this.stabilizers[pauli][iStab], activate);
            }
        }
    }

    getSyndrome() {
        let syndrome = {'X': [], 'Z': []};
        for (let pauli of ['X', 'Z']) {
            syndrome[pauli] = this.stabilizers[pauli].map(s => + s.isActivated)
        }

        return syndrome['Z'].concat(syndrome['X'])
    }

    insertError(qubit, pauli) {
        qubit.hasError[pauli] = !qubit.hasError[pauli];
    
        if (qubit.hasError['X'] || qubit.hasError['Z']) {    
            qubit.material.opacity = this.OPACITY.activatedQubit;

            if (qubit.hasError['X'] && qubit.hasError['Z']) {
                qubit.material.color.setHex(this.COLOR.errorY);
            }
            else if (qubit.hasError['X']) {
                qubit.material.color.setHex(this.COLOR.errorX);
            }
            else {
                qubit.material.color.setHex(this.COLOR.errorZ);
            }
        }
        else {
            qubit.material.opacity = this.opacityActivated ? this.OPACITY.minDeactivatedQubit : this.OPACITY.maxDeactivatedQubit;
            qubit.material.color.setHex(this.COLOR.deactivatedQubit);
        }
    
        this.updateStabilizers();
    }

    displayLogical(logical, pauli) {
        let index = this.currentIndexLogical[pauli]
        
        // Remove previous logical
        if (index != 0) {
            for(let i=0; i < logical[index - 1].length; i++) {
                if (logical[index - 1][i]) {
                    this.insertError(this.qubits[i], pauli)
                }
            }
        }
        // If index is equal to logical.length, we display no logicals
        if (index != logical.length) {
            // Insert new logical
            for(let i=0; i < logical[index].length; i++) {
                if (logical[index][i]) {
                    this.insertError(this.qubits[i], pauli)
                }
            }
        }

        this.currentIndexLogical[pauli] += 1
        this.currentIndexLogical[pauli] %= (logical.length + 1)
    }

    changeOpacity() {
        this.opacityActivated = !this.opacityActivated

        this.qubits.forEach(q => {
            if (!q.hasError['X'] && !q.hasError['Z']) {
                q.material.opacity = this.opacityActivated ? this.OPACITY.minDeactivatedQubit : this.OPACITY.maxDeactivatedQubit;
            }
            else {
                q.material.opacity = this.OPACITY.activatedQubit;
            }
        });

        ['X', 'Z'].forEach(pauli => {
            this.stabilizers[pauli].forEach(s => {
                if (!s.isActivated) {
                    s.material.opacity = this.opacityActivated ? 
                    this.OPACITY.minDeactivatedStabilizer[pauli] : this.OPACITY.maxDeactivatedStabilizer[pauli];
                }
                else {
                    s.material.opacity = this.OPACITY.activatedStabilizer[pauli];
                }
            });
        });
    }
}


class AbstractCubicCode extends AbstractCode {
    constructor(size, Hx, Hz, indices, scene) {
        super(size, Hx, Hz, indices, scene);

        this.rotatedPicture = false;

        this.vertices = [];
        this.faces = [];

        this.qubitIndex = indices['qubit'];
        this.vertexIndex = indices['vertex'];
        this.faceIndex = indices['face'];

        this.stabilizers['X'] = this.vertices;
        this.stabilizers['Z'] = this.faces;

        this.toggleStabFn['X'] = this.toggleVertex;
        this.toggleStabFn['Z'] = this.toggleFace;
        
        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.6,

            activatedStabilizer: {'X': 1, 'Z': 0.6},
            minDeactivatedStabilizer: {'X': 0.1, 'Z': 0},
            maxDeactivatedStabilizer: {'X': 0.6, 'Z': 0}
        }
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
        if (this.opacityActivated) {
            vertex.material.opacity = activate ? this.OPACITY.activatedStabilizer['X'] : this.OPACITY.minDeactivatedStabilizer['X'];
        }
        else {
            vertex.material.opacity = activate ? this.OPACITY.activatedStabilizer['X'] : this.OPACITY.maxDeactivatedStabilizer['X'];
        }
        let color = activate ? this.COLOR.activatedVertex : this.COLOR.deactivatedVertex;
        vertex.material.color.setHex(color);
    }
    
    toggleFace(face, activate) {
        face.isActivated = activate;
        if (this.opacityActivated) {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['Z'] : this.OPACITY.minDeactivatedStabilizer['Z'];
        }
        else {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['Z'] : this.OPACITY.maxDeactivatedStabilizer['Z'];
        }
    }

    buildQubit(x, y, z) {
        throw new Error('You have to implement the method buildEdge!');
    }

    buildVertex(x, y, z) {
        throw new Error('You have to implement the method buildVertex!');
    }

    buildFace(x, y, z) {
        throw new Error('You have to implement the method buildFace!');
    }

    build() {
        for (const [coord, index] of Object.entries(this.qubitIndex)) {
            let [x, y, z] = stringToArray(coord)
            this.buildQubit(x, y, z)
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


class AbstractRpCubicCode extends AbstractCode {
    constructor(size, Hx, Hz, indices, scene) {
        super(size, Hx, Hz, indices, scene);

        this.rotatedPicture = true;

        this.octahedrons = new Array(Hx.length);
        this.faces = new Array(Hz.length);

        this.qubitIndex = indices['qubit'];
        this.octahedronIndex = indices['vertex'];
        this.faceIndex = indices['face'];

        this.stabilizers['X'] = this.octahedrons;
        this.stabilizers['Z'] = this.faces;

        this.toggleStabFn['X'] = this.toggleOctahedron;
        this.toggleStabFn['Z'] = this.toggleFace;

        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.4,

            activatedStabilizer: {'X': 0.9, 'Z': 0.9},
            minDeactivatedStabilizer: {'X': 0.1, 'Z': 0.1},
            maxDeactivatedStabilizer: {'X': 0.3, 'Z': 0.3}
        }
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
            octa.material.opacity = activate ? this.OPACITY.activatedStabilizer['X'] : this.OPACITY.minDeactivatedStabilizer['X'];
        }
        else {
            octa.material.opacity = activate ? this.OPACITY.activatedStabilizer['X'] : this.OPACITY.maxDeactivatedStabilizer['X'];
        }
    }

    toggleFace(face, activate) {
        face.isActivated = activate;
        if (this.opacityActivated) {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['Z'] : this.OPACITY.minDeactivatedStabilizer['Z'];
        }
        else {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['Z'] : this.OPACITY.maxDeactivatedStabilizer['Z'];
        }
    }

    buildQubit(x, y, z) {
        throw new Error('You have to implement the method buildQubit!');
    }

    buildOctahedron(x, y, z) {
        throw new Error('You have to implement the method buildOctahedron!');
    }

    buildFace(x, y, z) {
        throw new Error('You have to implement the method buildFace!');
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
    }
}