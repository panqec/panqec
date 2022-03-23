export {AbstractCode, stringToArray};

function stringToArray(a) {
    return a.replace("[", "").replace("]", "").split(", ").map(Number)
}

class AbstractCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        this.H = {'X': Hx, 'Z': Hz};
        this.scene = scene;
        this.qubitIndex = qubitIndex;
        this.stabilizerIndex = stabilizerIndex
        this.stabilizerTypes = Object.keys(stabilizerIndex);
        this.qubitAxis = qubitAxis

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
            deactivatedVertex: 0xf2f2fc,
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
        console.log(index)

        let n_qubits = logical[index].length / 2;

        // If index is equal to logical.length, we display no logicals
        if (index != logical.length) {
            // Insert new logical
            for(let i=0; i < n_qubits; i++) {
                if (logical[index][i]) {
                    this.insertError(this.qubits[i], 'X')
                }
                if (logical[index][n_qubits+i]) {
                    this.insertError(this.qubits[i], 'Z')
                }
            }
        }

        this.currentIndexLogical[pauli] += 1
        this.currentIndexLogical[pauli] %= logical.length
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
