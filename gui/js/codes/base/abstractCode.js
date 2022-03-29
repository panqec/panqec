import nj from 'https://cdn.jsdelivr.net/npm/@d4c/numjs/build/module/numjs.min.js'

export {AbstractCode };

class AbstractCode {
    COLOR = {};
    OPACITY = {};
    SIZE = {radiusEdge: 0.07, radiusVertex: 0.14, lengthEdge: 1};

    constructor(size, H, qubitCoordinates, stabilizerCoordinates, qubitAxis, stabilizerType) {
        this.H = nj.array(H)
        this.qubitCoordinates = qubitCoordinates;
        this.stabilizerCoordinates = stabilizerCoordinates
        this.qubitAxis = qubitAxis
        this.stabilizerType = stabilizerType

        this.Lx = size[0];
        this.Ly = size[1];
        if (size.length > 2) {
            this.Lz = size[2];
        }

        this.n = this.qubitCoordinates.length // number of qubits
        this.m = this.stabilizerCoordinates.length // number of stabilizers

        this.opacityActivated = false;
        this.currentIndexLogical = {'X': 0, 'Z': 0};

        this.qubits = new Array(this.n);
        this.errors = nj.zeros(2*this.n);
        this.stabilizers = new Array(this.m);
    }

    updateStabilizers() {
        let syndrome = this.getSyndrome()

        for (let iStab=0; iStab < this.m; iStab++) {
            this.toggleStabilizer(this.stabilizers[iStab], syndrome[iStab]);
        }
    }

    getSyndrome() {
        let Hx = this.H.slice(null, [0, this.n]);
        let Hz = this.H.slice(null, [this.n, 2*this.n]);
        let ex = this.errors.slice([0, this.n]);
        let ez = this.errors.slice([this.n, 2*this.n]);

        let syndrome = nj.mod(nj.add(Hx.dot(ez), Hz.dot(ex)), 2);

        return syndrome.tolist();
    }

    insertError(qubit, pauli) {
        qubit.hasError[pauli] = !qubit.hasError[pauli];

        if (pauli == 'X' || pauli == 'Y') {
            this.errors.set(qubit.index, (this.errors.get(qubit.index) + 1) % 2)
        }
        if (pauli == 'Z' || pauli == 'Y') {
            this.errors.set(this.n + qubit.index, (this.errors.get(this.n + qubit.index) + 1) % 2)
        }

        if (qubit.hasError['X'] || qubit.hasError['Z']) {
            qubit.material.opacity = this.opacityActivated ? this.OPACITY.minActivatedQubit : this.OPACITY.maxActivatedQubit;

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
                q.material.opacity = this.opacityActivated ? this.OPACITY.minActivatedQubit : this.OPACITY.maxActivatedQubit;
            }
        });

        this.stabilizers.forEach(s => {
            let stabType = s.type
            if (!s.isActivated) {
                s.material.opacity = this.opacityActivated ?
                this.OPACITY.minDeactivatedStabilizer[stabType] : this.OPACITY.maxDeactivatedStabilizer[stabType];
            }
            else {
                s.material.opacity = this.opacityActivated ?
                this.OPACITY.minActivatedStabilizer[stabType] : this.OPACITY.maxActivatedStabilizer[stabType];
            }
        });
    }

    toggleStabilizer(stabilizer, activate, toggleVisible=false) {
        stabilizer.isActivated = activate;
        stabilizer.visible = activate || !toggleVisible;
        if (this.opacityActivated) {
            stabilizer.material.opacity = activate ? this.OPACITY.minActivatedStabilizer[stabilizer.type] : this.OPACITY.minDeactivatedStabilizer[stabilizer.type];
        }
        else {
            stabilizer.material.opacity = activate ? this.OPACITY.maxActivatedStabilizer[stabilizer.type] : this.OPACITY.maxDeactivatedStabilizer[stabilizer.type];
        }
        let color = activate ? this.COLOR.activatedStabilizer[stabilizer.type] : this.COLOR.deactivatedStabilizer[stabilizer.type];
        stabilizer.material.color.setHex(color);
    }

    buildQubit(index) {
        throw new Error('You have to implement the method buildQubit!');
    }

    buildStabilizer(index) {
        throw new Error('You have to implement the method buildStabilizer!');
    }

    build(scene) {
        let maxQubitCoordinates = {'x': 0, 'y': 0, 'z': 0};
        let maxStabCoordinates = {'x': 0, 'y': 0, 'z': 0};

        for (let index=0; index < this.n; index++) {
            let qubit = this.buildQubit(index);

            qubit.hasError = {'X': false, 'Z': false};

            qubit.index = index;
            qubit.location = this.qubitCoordinates[index];
            this.qubits[index] = qubit;

            maxQubitCoordinates['x'] = Math.max(qubit.position.x, maxQubitCoordinates['x']);
            maxQubitCoordinates['y'] = Math.max(qubit.position.x, maxQubitCoordinates['y']);
            maxQubitCoordinates['z'] = Math.max(qubit.position.x, maxQubitCoordinates['z']);

            scene.add(qubit);
        }
        for (let index=0; index < this.m; index++) {
            let stabilizer = this.buildStabilizer(index);

            stabilizer.isActivated = false;

            stabilizer.index = index;
            stabilizer.location = this.stabilizerCoordinates[index];
            stabilizer.type = this.stabilizerType[index];

            this.stabilizers[index] = stabilizer;

            maxStabCoordinates['x'] = Math.max(stabilizer.position.x, maxStabCoordinates['x']);
            maxStabCoordinates['y'] = Math.max(stabilizer.position.x, maxStabCoordinates['y']);
            maxStabCoordinates['z'] = Math.max(stabilizer.position.x, maxStabCoordinates['z']);

            scene.add(stabilizer);
        }

        var maxCoordinates = {'x': Math.max(maxStabCoordinates['x'], maxQubitCoordinates['x']),
                              'y': Math.max(maxStabCoordinates['y'], maxQubitCoordinates['y']),
                              'z': Math.max(maxStabCoordinates['z'], maxQubitCoordinates['z'])};

        var offset = {'x': maxCoordinates['x'] / 2,
                      'y': maxCoordinates['y'] / 2,
                      'z': maxCoordinates['z'] / 2};

        for (let qubit of this.qubits) {
            qubit.position.x -= offset['x'];
            qubit.position.y -= offset['y'];
            qubit.position.z -= offset['z'];
        }

        for (let stab of this.stabilizers) {
            stab.position.x -= offset['x'];
            stab.position.y -= offset['y'];
            stab.position.z -= offset['z'];
        }

        return maxCoordinates;
    }
}
