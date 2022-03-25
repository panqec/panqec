import nj from 'https://cdnjs.cloudflare.com/ajax/libs/numjs/0.16.1/numjs.min.js'

export {AbstractCode, stringToArray};

function stringToArray(a) {
    return a.replace("[", "").replace("]", "").split(", ").map(Number)
}

class AbstractCode {
    constructor(size, H, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        this.H = nj.array(H)
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

        this.n = H.shape[1] // number of qubits
        this.m = H.shape[0] // number of stabilizers

        this.opacityActivated = false;
        this.currentIndexLogical = {'X': 0, 'Z': 0};

        this.stabilizers = {};
        this.toggleStabFn = {};
        this.stabilizers = [];

        this.qubits = new Array(n);
        this.errors = nj.zeros(2*n);

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
        let syndrome = this.getSyndrome()

        for (let iStab=0; iStab < this.m; iStab++) {            
            this.toggleStabFn[stabType].call(this, this.stabilizers[iStab], syndrome[iStab]);
        }
    }

    getSyndrome() {
        let Hx = this.H.slice(null, [null, n])
        let Hz = this.H.slice(null, [n, null])
        let ex = this.errors.slice(null, [null, n])
        let ez = this.errors.slice(null, [n, null])

        let syndrome = (Hx.dot(ez) + Hz.dot(ex)) % 2

        return syndrome
    }

    insertError(qubit, pauli) {
        qubit.hasError[pauli] = !qubit.hasError[pauli];
        
        if (pauli == 'X' || pauli == 'Y') {
            this.errors[qubit.index] = (this.errors[qubit.index] + 1) % 2
        }
        if (pauli == 'Z' || pauli == 'Y') {
            this.errors[this.n + qubit.index] = (this.errors[qubit.index] + 1) % 2
        }

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

        // If index is equal to logical.length, we display no logicals
        if (index != logical.length) {
            // Insert new logical
            for(let i=0; i < this.n; i++) {
                if (logical[index][i]) {
                    this.insertError(this.qubits[i], 'X')
                }
                if (logical[index][this.n+i]) {
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
