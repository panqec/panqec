import { AbstractCode } from './abstractCode.js';

export {AbstractCubicCode, AbstractRpCubicCode};

function stringToArray(a) {
    return a.replace("[", "").replace("]", "").split(", ").map(Number)
}

class AbstractCubicCode extends AbstractCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene);

        this.rotatedPicture = false;

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
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene);

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
