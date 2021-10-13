export {AbstractCode, stringToArray};

class AbstractCode {
    constructor(Hx, Hz, scene) {
        this.H = {'X': Hx, 'Z': Hz};
        this.scene = scene;

        this.MIN_OPACITY = 0.1;
        this.MAX_OPACITY = 0.6;

        this.currentOpacity = this.MAX_OPACITY;
        
        this.stabilizers = {'X': [], 'Z': []};
        this.toggleStabFn = {'X': 0, 'Z': 0};
        this.qubits = new Array(Hx[0].length);
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
            qubit.material.transparent = false;
    
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
            qubit.material.transparent = true;
            qubit.material.opacity = this.currentOpacity;
            qubit.material.color.setHex(this.COLOR.deactivatedEdge);
        }
    
        this.updateStabilizers();
    }

    displayLogical(logical, pauli, indexLogical=0) {
        for(let i=0; i < logical[indexLogical].length; i++) {
            if (logical[indexLogical][i]) {
                this.insertError(this.qubits[i], pauli)
            }
        }
    }
}

function stringToArray(a) {
    return a.replace("[", "").replace("]", "").split(", ").map(Number)
}