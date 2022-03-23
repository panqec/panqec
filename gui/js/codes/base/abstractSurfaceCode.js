import { AbstractCode, stringToArray } from './abstractCode.js';

export {AbstractSurfaceCode, AbstractRpSurfaceCode};

class AbstractSurfaceCode extends AbstractCode {
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

        this.SIZE = {radiusEdge: 0.05 / (this.Lx/4), radiusVertex: 0.1 / (this.Lx/4), lengthEdge: 0.82 / (0.05 + this.Lx/4)};

        this.offset = {
            x: this.SIZE.lengthEdge * (this.Lx) / 2 - this.SIZE.lengthEdge/5,
            y: this.SIZE.lengthEdge * (this.Ly) / 2 - this.SIZE.lengthEdge/5
        };

        this.COLOR = {
            deactivatedVertex: 0xf2f2fc,
            activatedVertex: 0xf1c232,
            activatedFace: 0xf1c232, 
            deactivatedFace: 0xf1c232,
            deactivatedQubit: 0xffbcbc, 
            errorX: 0xFF4B3E,
            errorZ: 0x48BEFF,
            errorY: 0x058C42
        };
    }

    getIndexQubit(x, y) {
        let key = `[${x}, ${y}]`;
        return this.qubitIndex[key];
    }

    getIndexFace(x, y) {
        let key = `[${x}, ${y}]`;
        return this.faceIndex[key];
    }
 
    getIndexVertex(x, y) {
        let key = `[${x}, ${y}]`;
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
        let color = activate ? this.COLOR.activatedFace : this.COLOR.deactivatedFace;
        face.material.color.setHex(color);
    }

    buildQubit(x, y) {
        throw new Error('You have to implement the method buildQubit!');
    }

    buildVertex(x, y) {
        throw new Error('You have to implement the method buildVertex!');
    }

    buildFace(x, y) {
        throw new Error('You have to implement the method buildFace!');
    }

    build() {
        for (const [coord, index] of Object.entries(this.qubitIndex)) {
            let [x, y] = stringToArray(coord)
            this.buildQubit(x, y)
        }
        for (const [coord, index] of Object.entries(this.vertexIndex)) {
            let [x, y] = stringToArray(coord)
            this.buildVertex(x, y)
        }
        for (const [coord, index] of Object.entries(this.faceIndex)) {
            let [x, y] = stringToArray(coord)
            this.buildFace(x, y)
        }
    }
}


class AbstractRpSurfaceCode extends AbstractSurfaceCode {
    constructor(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene) {
        super(size, Hx, Hz, qubitIndex, stabilizerIndex, qubitAxis, scene);

        this.rotatedPicture = true;

        this.OPACITY = {
            activatedQubit: 1,
            minDeactivatedQubit: 0.1,
            maxDeactivatedQubit: 0.6,

            activatedStabilizer: {'vertex': 0.9, 'face': 0.9},
            minDeactivatedStabilizer: {'vertex': 0.1, 'face': 0.1},
            maxDeactivatedStabilizer: {'vertex': 0.2, 'face': 0.2}
        }

        // this.SIZE = {radiusEdge: 0.05 / (this.Lx/4), radiusVertex: 0.1 / (this.Lx/4), lengthEdge: 0.9 / (this.Lx/4)};

        this.COLOR = {
            activatedVertex: 0xfabc2a, 
            deactivatedVertex: 0xFAFAC6,
            activatedFace: 0xFA824C, 
            deactivatedFace: 0xe79e90,
            activatedQubit: 0xffbcbc,
            deactivatedQubit: 0xffbcbc, 
            errorX: 0xFF4B3E,
            errorZ: 0x4381C1,
            errorY: 0x058C42
        };
    }
}
