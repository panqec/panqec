import * as THREE from 'https://cdn.skypack.dev/three@v0.130.1';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/controls/OrbitControls.js';
import { OutlineEffect } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/effects/OutlineEffect.js';
import { GUI } from 'https://cdn.skypack.dev/three@0.130.0/examples/jsm/libs/dat.gui.module';
import nj from 'https://cdn.jsdelivr.net/npm/@d4c/numjs/build/module/numjs.min.js'

import { TopologicalCode } from './topologicalCode.js';

export { Interface }

const defaultParams = {
    dimension: 2,
    codeName: 'Toric 2D',
    Lx: 6,
    Ly: 6,
    Lz: 6,
    errorModel: 'Depolarizing',
    errorProbability: 0.1,
    decoder: 'BP-OSD',
    noiseDeformationName: 'None',
    codeDeformationName: 'None',
    max_bp_iter: 10,
    alpha: 0.4,
    beta: 0,
    channel_update: false,
    rotated: false,
    coprime: false
};

const defaultColors = {
    background: 0xffffff
};

const defaultKeyCode = {
    'decode': 68, // 'd'
    'random': 82, // 'r'
    'remove': 8,  // 'backspace'
    'opacity': 79, // 'o'
    'x-logical': 88, // 'x'
    'z-logical': 90, // 'z'
    'x-error': 17, // 'ctrl'
    'z-error': 16, // 'shift'
    'contract': 67, // 'c'
    'check-logical-error': 76  // 'l'
};

class Interface {
    constructor(
        params=defaultParams,
        colors=defaultColors,
        keycode=defaultKeyCode,
        url='',
        containerId=''
    ) {
        this.params = Object.assign({}, defaultParams);
        this.colors = Object.assign({}, defaultColors);
        this.keycode = Object.assign({}, defaultKeyCode);

        Object.entries(params).forEach(entry => {
            const [key, value] = entry;
            this.params[key] = value;
        }, this);

        Object.entries(colors).forEach(entry => {
            const [key, value] = entry;
            this.colors[key] = value;
        }, this);

        Object.entries(keycode).forEach(entry => {
            const [key, value] = entry;
            this.keycode[key] = value;
        }, this);

        this.url = url;
        if (containerId === '') {
            this.container = document.body;
        }
        else {
            this.container = document.getElementById(containerId);
        }

        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        if (this.height == 0) {
            this.height = window.innerHeight;
        }

        this.fullWindow = (this.width == window.innerWidth && this.height == window.innerHeight);

        this.buttons = {
            'decode': this.decode,
            'addErrors': this.addRandomErrors
        };
    }

    async init() {
        if (this.params.dimension == 2) {
            this.buildScene2D();
        }
        else {
            this.buildScene3D();
        }
        await this.buildCode();

        if (this.params.dimension == 3) {
            this.controls.update();
        }

        this.animate()
    }

    buildScene2D() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.colors.background );

        // Camera
        this.camera = new THREE.PerspectiveCamera( 10, this.width / this.height, 0.1, 1000 );
        this.camera.position.z = 25;
        this.camera.position.y = 0;
        this.camera.position.x = 0;

        const dirLight1 = new THREE.DirectionalLight( 0xffffff );
        dirLight1.position.set( 1, 1, 1 );
        this.scene.add( dirLight1 );

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // var canvasDict = this.canvas === null ? {} : {canvas: this.canvas};

        this.renderer = new THREE.WebGLRenderer();

        this.renderer.setSize(this.width, this.height);
        this.container.appendChild(this.renderer.domElement);

        document.addEventListener("keydown", this.onDocumentKeyDown.bind(this), false);
        document.addEventListener('mousedown', this.onDocumentMouseDown.bind(this), false);
        if (this.fullWindow) {
            window.addEventListener('resize', this.onWindowResize.bind(this), false);
        }
            window.addEventListener("contextmenu", e => e.preventDefault());
    }

    buildScene3D() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.colors.background);

        this.camera = new THREE.PerspectiveCamera( 75, this.width / this.height, 0.1, 1000 );

        this.camera.position.x = 5;
        this.camera.position.y = 3.5;
        this.camera.position.z = 7;

        const dirLight2 = new THREE.DirectionalLight( 0x002288 );
        dirLight2.position.set( - 1, - 1, - 1 );
        this.scene.add( dirLight2 );

        const dirLight3 = new THREE.DirectionalLight(0x002288);
        dirLight3.position.set(4, 4, 4);
        this.scene.add( dirLight3 );

        const pointLight = new THREE.PointLight(0xffffff, 1, 0, 1);
        this.scene.add(pointLight);
        this.camera.add(pointLight);
        this.scene.add(this.camera);

        const ambientLight = new THREE.AmbientLight(0x222222);
        this.scene.add( ambientLight );

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        this.renderer = new THREE.WebGLRenderer();
        this.renderer.setSize(this.width, this.height);
        this.container.appendChild(this.renderer.domElement);

        this.effect = new OutlineEffect(this.renderer);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.maxPolarAngle = THREE.Math.degToRad(270);
        this.controls.update();

        document.addEventListener("keydown", this.onDocumentKeyDown.bind(this), false);
        document.addEventListener( 'mousedown', this.onDocumentMouseDown.bind(this), false );
        window.addEventListener('resize', this.onWindowResize.bind(this), false);
    }

    async buildCode() {
        let data = await this.getCodeData();
        let H = data['H'];
        let qubits = data['qubits'];
        let stabilizers = data['stabilizers'];
        let logical_z = data['logical_z'];
        let logical_x = data['logical_x'];
        let Lx = this.params.Lx;
        let Ly = this.params.Ly;
        let Lz = this.params.Lz;

        if (this.params.dimension == 2) {
            var size = [Lx, Ly];
        }
        else {
            var size = [Lx, Ly, Lz];
        }

        this.code = new TopologicalCode(size, H, qubits, stabilizers);
        this.code.logical_x = logical_x;
        this.code.logical_z = logical_z;
        var maxCoordinates = this.code.build(this.scene);

        if (this.params.dimension == 2) {
            var fov = (this.camera.fov * Math.PI) / 180;
            this.camera.position.z = Math.max(maxCoordinates['x'], maxCoordinates['y']) / (2*Math.tan(fov/2)) + 10;
        }
    }

    async changeLatticeSize() {
        this.params.Lx = parseInt(this.params.Lx)
        this.params.Ly = this.params.Lx;
        this.params.Lz = this.params.Lx;

        if (this.params.coprime)
            this.params.Lx += 1;

        this.code.qubits.forEach(q => {
            q.material.dispose();
            q.geometry.dispose();

            this.scene.remove(q);
        });

        this.code.stabilizers.forEach(s => {
            s.material.dispose();
            s.geometry.dispose();

           this.scene.remove(s);
        });

        await this.updateMenu();
        this.buildCode();
    }

    async getCodeData() {
        let response = await fetch(this.url + '/code-data', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({
                'Lx': this.params.Lx,
                'Ly': this.params.Ly,
                'Lz': this.params.Lz,
                'code_name': this.params.codeName,
                'code_deformation_name': this.params.codeDeformationName,
                'rotated_picture': this.params.rotated
            })
        });

        let data  = await response.json();

        return data;
    }

    async getCodeNames() {
        let response = await fetch(this.url + '/code-names', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({'dimension': this.params.dimension})
        });

        let data  = await response.json();

        return data;
    }

    async getDecoderNames() {
        let response = await fetch(this.url + '/decoder-names', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({'code_name': this.params.codeName})
        });

        let data  = await response.json();

        return data;
    }

    async getDeformationNames() {
        let response = await fetch(this.url + '/deformation-names', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({'code_name': this.params.codeName})
        });

        let data  = await response.json();

        return data;
    }

    async buildMenu() {
        this.menu = new GUI({width: 300});
        const codeFolder = this.menu.addFolder('Code')

        var codes = await this.getCodeNames();

        codeFolder.add(this.params, 'codeName', codes).name('Code type').onChange(this.changeLatticeSize.bind(this));
        codeFolder.add(this.params, 'rotated').name('Rotated picture').onChange(this.changeLatticeSize.bind(this));
        codeFolder.add(this.params, 'coprime').name('Coprime dimensions').onChange(this.changeLatticeSize.bind(this));
        codeFolder.add(this.params, 'Lx', {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                                    '8': 8, '9': 9, '10':10, '11':11, '12': 12}).name('Lattice size').onChange(this.changeLatticeSize.bind(this));
        codeFolder.open();

        this.updateMenu();
    }

    async updateMenu() {
        // Clifford-deformation part
        var deformationNames = await this.getDeformationNames();
        deformationNames = ['None'].concat(deformationNames)

        if (!deformationNames.includes(this.params.codeDeformationName)) {
            this.params.codeDeformationName = 'None';
        }
        if (!deformationNames.includes(this.params.noiseDeformationName)) {
            this.params.noiseDeformationName = 'None';
        }

        var codeFolder = this.menu.__folders['Code'];

        codeFolder.__controllers.forEach(controller => {
            if (controller.property == 'codeDeformationName') {
                controller.remove();
            }
        });

        codeFolder.add(this.params, 'codeDeformationName', deformationNames).name('Clifford deformation').onChange(this.changeLatticeSize.bind(this));

        if ('Error Model' in this.menu.__folders) {
            this.menu.removeFolder(this.menu.__folders['Error Model']);
        }

        const errorModelFolder = this.menu.addFolder('Error Model')
        errorModelFolder.add(this.params, 'errorModel',
            {'Pure X': 'Pure X', 'Pure Y': 'Pure Y', 'Pure Z': 'Pure Z', 'Depolarizing': 'Depolarizing'}
        ).name('Model');
        errorModelFolder.add(this.params, 'errorProbability', 0, 0.5).name('Probability');
        errorModelFolder.add(this.params, 'noiseDeformationName', deformationNames).name('Clifford deformation').onChange(this.changeLatticeSize.bind(this));
        errorModelFolder.add(this.buttons, 'addErrors').name('▶ Add errors (r)');
        errorModelFolder.open();

        // Decoder part
        if ('Decoder' in this.menu.__folders) {
            this.menu.removeFolder(this.menu.__folders['Decoder']);
        }

        var decoders = await this.getDecoderNames();

        if (!decoders.includes(this.params.decoder)) {
            this.params.decoder = decoders[0];
        }

        const decoderFolder = this.menu.addFolder('Decoder');

        decoderFolder.add(this.params, 'decoder', decoders).name('Decoder');
        decoderFolder.add(this.params, 'max_bp_iter', 1, 1000, 1).name('Max iterations (BP)');
        decoderFolder.add(this.params, 'channel_update').name('Channel update (BP)');
        decoderFolder.add(this.params, 'alpha', 0.01, 2, 0.01).name('Alpha (MBP)');
        decoderFolder.add(this.params, 'beta', 0, 2, 0.01).name('Beta (MBP)');
        decoderFolder.add(this.buttons, 'decode').name("▶ Decode (d)");
        decoderFolder.open();
    }

    toggleInstructions() {
        var closingCross = document.getElementById('closingCross');
        var instructions = document.getElementById('instructions');

        if (instructions.style.visibility == 'hidden') {
            instructions.style.visibility = 'visible';
            closingCross.innerHTML = "<a href='#'>× Instructions</a>";

        }
        else {
            instructions.style.visibility = 'hidden';
            closingCross.innerHTML = "<a href='#'>🔽 Instructions</a>";
        }
    }

    buildInstructions() {
        var closingCross = document.createElement('div');
        closingCross.id = 'closingCross';
        closingCross.innerHTML = "<a href='#'>× Instructions</a>";
        closingCross.onclick = this.toggleInstructions;

        var instructions = document.createElement('div');
        instructions.id = 'instructions';
        if (this.params.dimension == 3) {
            instructions.innerHTML =
            "\
                <table style='border-spacing: 10px'>\
                <tr><td><b>Ctrl click</b></td><td>X error</td></tr>\
                <tr><td><b>Shift click</b></td><td>Z error</td></tr>\
                <tr><td><b>Backspace</b></td><td>Remove errors</td></tr>\
                <tr><td><b>R</b></td><td>Random errors</td></tr>\
                <tr><td><b>D</b></td><td>Decode</td></tr>\
                <tr><td><b>O</b></td><td>Toggle Opacity</td></tr>\
                <tr><td><b>Z</b></td><td>Logical Z</td></tr>\
                <tr><td><b>X</b></td><td>Logical X</td></tr>\
                </table>\
            ";
        }
        else {
            instructions.innerHTML =
            "\
                <table style='border-spacing: 10px'>\
                <tr><td><b>Ctrl click</b></td><td>X error</td></tr>\
                <tr><td><b>Shift click</b></td><td>Z error</td></tr>\
                <tr><td><b>Backspace</b></td><td>Remove errors</td></tr>\
                <tr><td><b>R</b></td><td>Random errors</td></tr>\
                <tr><td><b>D</b></td><td>Decode</td></tr>\
                <tr><td><b>O</b></td><td>Toggle Opacity</td></tr>\
                <tr><td><b>Z</b></td><td>Logical Z</td></tr>\
                <tr><td><b>X</b></td><td>Logical X</td></tr>\
                </table>\
            ";
        }
        document.body.appendChild(instructions);
        document.body.appendChild(closingCross);
    }

    buildReturnArrow() {
        var returnArrow = document.createElement('div');
        returnArrow.id = 'returnArrow'
        returnArrow.innerHTML = "<a href='/'>❮</a>"

        document.body.appendChild(returnArrow);
    }

    async getCorrection(syndrome) {
        let response = await fetch(this.url + '/decode', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({
                'Lx': this.params.Lx,
                'Ly': this.params.Ly,
                'Lz': this.params.Lz,
                'p': this.params.errorProbability,
                'max_bp_iter': this.params.max_bp_iter,
                'alpha': this.params.alpha,
                'beta': this.params.beta,
                'channel_update': this.params.channel_update,
                'syndrome': syndrome,
                'noise_deformation_name': this.params.noiseDeformationName,
                'decoder': this.params.decoder,
                'error_model': this.params.errorModel,
                'code_name': this.params.codeName,
                'code_deformation_name': this.params.codeDeformationName
            })
        });

        let data  = await response.json();

        return data
    }

    async checkLogicalError() {
        let response = await fetch(this.url + '/check-logical-error', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({
                'error': this.code.errors,
                'Lx': this.params.Lx,
                'Ly': this.params.Ly,
                'Lz': this.params.Lz,
                'code_name': this.params.codeName,
                'code_deformation_name': this.params.codeDeformationName
            })
        });

        let data  = await response.json();

        return data
    }

    async getRandomErrors() {
        let response = await fetch(this.url + '/new-errors', {
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST',
            body: JSON.stringify({
                'Lx': this.params.Lx,
                'Ly': this.params.Ly,
                'Lz': this.params.Lz,
                'p': this.params.errorProbability,
                'noise_deformation_name': this.params.noiseDeformationName,
                'error_model': this.params.errorModel,
                'code_name': this.params.codeName,
                'code_deformation_name': this.params.codeDeformationName
            })
        });

        let data  = await response.json();

        return data;
    }

    async addRandomErrors() {
        let errors = new Array(2*this.code.n).fill(0);

        if (this.params.noiseDeformationName == 'None') {
            if (this.params.errorModel.includes('Pure')) {
                for(var i=0; i < this.code.n; i++) {
                    let err = +(Math.random() < this.params.errorProbability)
                    if (this.params.errorModel.includes('X') || this.params.errorModel.includes('Y'))
                        errors[i] = err
                    if (this.params.errorModel.includes('Z') || this.params.errorModel.includes('Y'))
                        errors[this.code.n+i] = err
                }
            }
        }

        if (errors.length == 0)
            errors = await this.getRandomErrors()

        let n = this.code.n;
        this.code.qubits.forEach((q, i) => {
            if (errors[i]) {
                this.code.insertError(q, 'X', false);
            }
            if (errors[n+i]) {
                this.code.insertError(q, 'Z', false);
            }
        });
        this.code.updateStabilizers();
    }

    removeAllErrors() {
        this.code.qubits.forEach(q => {
            ['X', 'Z'].forEach(errorType => {
                if (q.hasError[errorType]) {
                    this.code.insertError(q, errorType, false);
                }
            })
        });
        this.code.updateStabilizers();
    }

    contractError() {
        for(var i=0; i < this.code.H.shape[0]; i++) {
            var newError = nj.mod(this.code.errors.add(this.code.H.pick(i, null)), 2);
            var doInsert = false;
            if (newError.sum() < this.code.errors.sum())
                doInsert = true;
            if (newError.sum() == this.code.errors.sum())
                doInsert = (Math.random() < 0.5);

            if (doInsert) {
                for(var j=0; j < 2*this.code.n; j++) {
                    if (this.code.H.get(i, j) == 1) {
                        if (j < this.code.n)
                            this.code.insertError(this.code.qubits[j], 'X', false)
                        else
                            this.code.insertError(this.code.qubits[j-this.code.n], 'Z', false)
                    }
                }
            }
        }
        console.log("New weight", this.code.errors.sum())
        this.code.updateStabilizers();
    }

    async decode() {
        let syndrome = this.code.getSyndrome();
        let correction = await this.getCorrection(syndrome)

        correction['x'].forEach((c,i) => {
            if(c) {
                this.code.insertError(this.code.qubits[i], 'X', false)
            }
        });
        correction['z'].forEach((c,i) => {
            if(c) {
                this.code.insertError(this.code.qubits[i], 'Z', false)
            }
        });
        this.code.updateStabilizers();
    }

    onDocumentMouseDown(event) {
        var canvasBound = this.renderer.getContext().canvas.getBoundingClientRect();

        this.mouse.x = ( (event.clientX  - canvasBound.left) / this.width ) * 2 - 1;
        this.mouse.y = - ( (event.clientY - canvasBound.top) / this.height ) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);

        this.intersects = this.raycaster.intersectObjects(this.code.qubits);
        if (this.intersects.length == 0) return;

        let selectedQubit = this.intersects[0].object;

        if (event.button == 0) {
            var x = selectedQubit.location[0];
            var y = selectedQubit.location[1];
            var z = selectedQubit.location[2];

            var correctMouseKey = function(keycode) {
                return (
                    ((keycode == 17) && event.ctrlKey) ||
                    ((keycode == 16) && event.shiftKey) ||
                    keycode == 0
                )
            }

            if (correctMouseKey(this.keycode['x-error']))
            {
                console.log('Selected qubit', selectedQubit.index, 'at', x, y, z);
                this.code.insertError(selectedQubit, 'X', true);
            }
            if (correctMouseKey(this.keycode['z-error'])) {
                console.log('Selected qubit', selectedQubit.index, 'at', x, y, z);
                this.code.insertError(selectedQubit, 'Z', true);
            }
        }
    }

    onDocumentKeyDown(event) {
        var keyCode = event.which;

        if (keyCode == this.keycode['decode']) {
            this.decode()
        }

        else if (keyCode == this.keycode['random']) {
            this.addRandomErrors();
        }

        else if (keyCode == this.keycode['remove']) {
            this.removeAllErrors();
        }

        else if (keyCode == this.keycode['opacity']) {
            this.code.changeOpacity();
        }

        else if (keyCode == this.keycode['x-logical']) {
            // this.removeAllErrors();
            this.code.displayLogical(this.code.logical_x, 'X');
        }

        else if (keyCode == this.keycode['z-logical']) {
            // this.removeAllErrors();
            this.code.displayLogical(this.code.logical_z, 'Z');
        }
        else if (keyCode == this.keycode['contract']) {
            this.contractError();
        }
        else if (keyCode == this.keycode['check-logical-error']) {
            this.checkLogicalError();
        }
    };

    onWindowResize(){
        this.width = window.innerWidth;
        this.height = window.innerHeight;

        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize( window.innerWidth, window.innerHeight );
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));

        this.raycaster.setFromCamera(this.mouse, this.camera);

        if (this.params.dimension == 3) {
            this.controls.update();
            this.effect.render(this.scene, this.camera);
        }
        else {
            this.renderer.render(this.scene, this.camera);
        }
    }
}
