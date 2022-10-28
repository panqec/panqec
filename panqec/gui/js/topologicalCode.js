import nj from 'https://cdn.jsdelivr.net/npm/@d4c/numjs/build/module/numjs.min.js'

import { create_shape } from './shapes.js'

export { TopologicalCode };

class TopologicalCode {
    constructor(size, H, qubitData, stabilizerData) {
        this.H = nj.array(H)
        this.qubitData = qubitData;
        this.stabilizerData = stabilizerData;

        this.Lx = size[0];
        this.Ly = size[1];
        if (size.length > 2) {
            this.Lz = size[2];
        }

        this.n = this.qubitData.length // number of qubits
        this.m = this.stabilizerData.length // number of stabilizers

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

        var opacityLevel = this.opacityActivated ? 'min' : 'max';
        if (qubit.hasError['X'] || qubit.hasError['Z']) {
            qubit.material.opacity = qubit.opacity['activated'][opacityLevel]

            if (qubit.hasError['X'] && qubit.hasError['Z']) {
                qubit.material.color.setHex(qubit.color['Y']);
            }
            else if (qubit.hasError['X']) {
                qubit.material.color.setHex(qubit.color['X']);
            }
            else {
                qubit.material.color.setHex(qubit.color['Z']);
            }
        }
        else {
            qubit.material.opacity = qubit.opacity['deactivated'][opacityLevel]
            qubit.material.color.setHex(qubit.color['I']);
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

        var opacityLevel = this.opacityActivated ? 'min' : 'max';

        this.qubits.forEach(q => {
            let activatedStr = (q.hasError['X'] || q.hasError['Z']) ? 'activated' : 'deactivated';
            q.material.opacity = q.opacity[activatedStr][opacityLevel];
        });

        this.stabilizers.forEach(s => {
            let activatedStr = s.isActivated ? 'activated' : 'deactivated';
            s.material.opacity = s.opacity[activatedStr][opacityLevel];
        });
    }

    toggleStabilizer(stabilizer, activate) {
        stabilizer.isActivated = activate;
        var activateStr = activate ? 'activated' : 'deactivated';
        var opacityLevel = this.opacityActivated ? 'min' : 'max'

        stabilizer.material.opacity = stabilizer.opacity[activateStr][opacityLevel];
        stabilizer.material.color.setHex(stabilizer.color[activateStr]);

        // Prevents a weird glitch with opacity-0 objects
        stabilizer.visible = (stabilizer.material.opacity != 0)

        // For objects with a wireframe
        if (stabilizer.children.length > 0 && stabilizer.children[0].type == 'LineSegments') {
            var wireframe = stabilizer.children[0]
            wireframe.material.opacity = stabilizer.material.opacity;
        }
    }

    buildQubit(index) {
        var qubit = create_shape[this.qubitData[index]['object']](this.qubitData[index]['location'], this.qubitData[index]['params']);

        qubit.index = index
        qubit.color = this.qubitData[index]['color']
        qubit.opacity = this.qubitData[index]['opacity']

        return qubit
    }

    buildStabilizer(index) {
        var stabilizer = create_shape[this.stabilizerData[index]['object']](this.stabilizerData[index]['location'],
                                                                     this.stabilizerData[index]['params']);

        stabilizer.index = index;
        stabilizer.color = this.stabilizerData[index]['color']
        stabilizer.opacity = this.stabilizerData[index]['opacity']
        stabilizer.type = this.stabilizerData[index]['type']

        return stabilizer
    }

    build(scene) {
        let maxQubitCoordinates = {'x': 0, 'y': 0, 'z': 0};
        let maxStabCoordinates = {'x': 0, 'y': 0, 'z': 0};
        var opacityLevel = this.opacityActivated ? 'min' : 'max'

        for (let index=0; index < this.n; index++) {
            let qubit = this.buildQubit(index);

            qubit.material.color.setHex(qubit.color['I']);
            qubit.material.opacity = qubit.opacity['deactivated'][opacityLevel]

            qubit.hasError = {'X': false, 'Z': false};

            qubit.index = index;
            qubit.location = this.qubitData[index]['location'];
            this.qubits[index] = qubit;

            maxQubitCoordinates['x'] = Math.max(qubit.position.x, maxQubitCoordinates['x']);
            maxQubitCoordinates['y'] = Math.max(qubit.position.y, maxQubitCoordinates['y']);
            maxQubitCoordinates['z'] = Math.max(qubit.position.z, maxQubitCoordinates['z']);

            scene.add(qubit);
        }
        for (let index=0; index < this.m; index++) {
            let stabilizer = this.buildStabilizer(index);

            var stabType = this.stabilizerData[index]['type'];
            stabilizer.material.color.setHex(stabilizer.color['deactivated']);
            stabilizer.material.opacity = stabilizer.opacity['deactivated'][opacityLevel];

            // Prevents a weird glitch with opacity-0 objects
            stabilizer.visible = (stabilizer.material.opacity != 0)

            // For objects with a wireframe
            if (stabilizer.children.length > 0 && stabilizer.children[0].type == 'LineSegments') {
                var wireframe = stabilizer.children[0]
                wireframe.material.opacity = stabilizer.material.opacity;
            }

            stabilizer.isActivated = false;

            stabilizer.index = index;
            stabilizer.location = this.stabilizerData[index]['location'];
            stabilizer.type = stabType;

            this.stabilizers[index] = stabilizer;

            maxStabCoordinates['x'] = Math.max(stabilizer.position.x, maxStabCoordinates['x']);
            maxStabCoordinates['y'] = Math.max(stabilizer.position.x, maxStabCoordinates['y']);
            maxStabCoordinates['z'] = Math.max(stabilizer.position.x, maxStabCoordinates['z']);

            scene.add(stabilizer);
        }

        var maxCoordinates = {'x': Math.max(maxStabCoordinates['x'], maxQubitCoordinates['x'])+1,
                              'y': Math.max(maxStabCoordinates['y'], maxQubitCoordinates['y'])+1,
                              'z': Math.max(maxStabCoordinates['z'], maxQubitCoordinates['z'])+1};

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
