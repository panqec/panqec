export {AbstractCode, AbstractCubicCode, AbstractRpCubicCode, stringToArray};

function stringToArray(a) {
    return a.replace("[", "").replace("]", "").split(", ").map(Number)
}

class AbstractCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, scene) {
        this.H = {'X': Hx, 'Z': Hz};
        this.scene = scene;
        this.stabilizerIndex = stabilizerIndex
        this.stabilizerTypes = Object.keys(stabilizerIndex);

        this.Lx = size[0];
        this.Ly = size[1];
        if (size.length > 2) {
            this.Lz = size[2];
        }

        this.opacityActivated = false;
        this.currentIndexLogical = {'X': 0, 'Z': 0};
        
        this.stabilizers = {};
        this.toggleStabFn = {};
        this.stabilizers = [];
        
        this.qubits = new Array(Hx[0].length);
        this.COLOR = {
            deactivatedVertex: 0xf2f28c,
            activatedVertex: 0xf1c232,
            deactivatedEdge: 0xffbcbc,
            deactivatedQubit: 0xffbcbc,
            activatedFace: 0xf1c232, 
            activatedOctahedron: 0xfa7921, 
            errorX: 0xFF4B3E,
            errorZ: 0x4381C1,
            errorY: 0x058C42
        };

        this.SIZE = {radiusEdge: 0.05, radiusVertex: 0.1, lengthEdge: 1};
        let length = this.SIZE.lengthEdge;
        this.offset = {x: Math.SQRT2 * length*this.Lx / 2, y: Math.SQRT2 * length*this.Ly / 2, z: length*this.Lz / 2};
    }

    updateStabilizers() {
        let nQubitErrors;
        for (let iStab=0; iStab < this.H['X'].length; iStab++) {
            nQubitErrors = 0
            for (let pauli of ['X' , 'Z']) {
                for (let iQubit=0; iQubit < this.H[pauli][0].length; iQubit++) {
                    if (this.H[pauli][iStab][iQubit] == 1) {
                        if (this.qubits[iQubit].hasError[pauli]) {
                            nQubitErrors += 1
                        }
                    }
                }
            }
            let stabType = this.stabilizers[iStab].type
            let activate = (nQubitErrors % 2 == 1);
            this.toggleStabFn[stabType].call(this, this.stabilizers[iStab], activate);
        }
    }
    
    getSyndrome() {
        let syndrome = [];
        syndrome = this.stabilizers.map(s => + s.isActivated)
        return syndrome
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

        this.stabilizers.forEach(s => {
            let stabType = s.type
            if (!s.isActivated) {
                s.material.opacity = this.opacityActivated ? 
                this.OPACITY.minDeactivatedStabilizer[stabType] : this.OPACITY.maxDeactivatedStabilizer[stabType];
            }
            else {
                s.material.opacity = this.OPACITY.activatedStabilizer[stabType];
            }
        });
    }
}


class AbstractCubicCode extends AbstractCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, scene);

        this.rotatedPicture = false;

        this.qubitIndex = qubitIndex;
        this.vertexIndex = stabilizerIndex['vertex'];
        this.faceIndex = stabilizerIndex['face'];

        this.stabilizers = new Array(Hx.length);

        this.toggleStabFn['vertex'] = this.toggleVertex;
        this.toggleStabFn['face'] = this.toggleFace;
        
        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.6,

            activatedStabilizer: {'vertex': 1, 'face': 0.6},
            minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0},
            maxDeactivatedStabilizer: {'vertex': 0.6, 'face': 0}
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
            vertex.material.opacity = activate ? this.OPACITY.activatedStabilizer['vertex'] : this.OPACITY.minDeactivatedStabilizer['vertex'];
        }
        else {
            vertex.material.opacity = activate ? this.OPACITY.activatedStabilizer['vertex'] : this.OPACITY.maxDeactivatedStabilizer['vertex'];
        }
        let color = activate ? this.COLOR.activatedVertex : this.COLOR.deactivatedVertex;
        vertex.material.color.setHex(color);
    }
    
    toggleFace(face, activate) {
        face.isActivated = activate;
        if (this.opacityActivated) {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['face'] : this.OPACITY.minDeactivatedStabilizer['face'];
        }
        else {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['face'] : this.OPACITY.maxDeactivatedStabilizer['face'];
        }
    }

    buildQubit(x, y, z) {
        throw new Error('You have to implement the method buildQubit!');
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
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, scene);

        this.rotatedPicture = true;

        this.octahedrons = new Array(Hx.length);
        this.faces = new Array(Hz.length);

        this.qubitIndex = qubitIndex;
        this.octahedronIndex = stabilizerIndex['vertex'];
        this.faceIndex = stabilizerIndex['face'];

        this.stabilizers = new Array(Hx.length);

        this.toggleStabFn['octahedron'] = this.toggleOctahedron;
        this.toggleStabFn['face'] = this.toggleFace;

        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.4,

            activatedStabilizer: {'octahedron': 0.9, 'face': 0.9},
            minDeactivatedStabilizer: {'octahedron': 0.1, 'face': 0.1},
            maxDeactivatedStabilizer: {'octahedron': 0.3, 'face': 0.3}
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
            octa.material.opacity = activate ? this.OPACITY.activatedStabilizer['octahedron'] : this.OPACITY.minDeactivatedStabilizer['octahedron'];
        }
        else {
            octa.material.opacity = activate ? this.OPACITY.activatedStabilizer['octahedron'] : this.OPACITY.maxDeactivatedStabilizer['octahedron'];
        }
    }

    toggleFace(face, activate) {
        face.isActivated = activate;
        if (this.opacityActivated) {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['face'] : this.OPACITY.minDeactivatedStabilizer['face'];
        }
        else {
            face.material.opacity = activate ? this.OPACITY.activatedStabilizer['face'] : this.OPACITY.maxDeactivatedStabilizer['face'];
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