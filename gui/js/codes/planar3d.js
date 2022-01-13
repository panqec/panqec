import { ToricCode3D } from './toric3d.js';

export {PlanarCode3D};

class PlanarCode3D extends ToricCode3D {
    constructor(size, Hx, Hz, indices, scene) {
        super(size, Hx, Hz, indices, scene);
    }
}
