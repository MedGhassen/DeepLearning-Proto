/**
 * dA.ts Denoising Autoencoders (dA)
 * Mohamed Ghassen Brahim <mohamed.ghassen.brahim@gmail.com>
 */
var utils = require("../utils/utils.ts");
var dA = (function () {
    function dA(size, n_v, n_h, w, hb, vb) {
        this.N = size;
        this.n_visible = n_v;
        this.n_hidden = n_h;
        if (w.length == 0) {
            this.w = new Array(this.n_hidden);
            for (var i = 0; i < this.n_hidden; i++) {
                this.w[i] = new Array(this.n_visible);
            }
            var a = 1 / this.n_visible;
            for (var i = 0; i < this.n_hidden; i++) {
                for (var j = 0; j < this.n_visible; j++) {
                    this.w[i][j] = utils.uniform(-a, a);
                }
            }
        }
        else {
            this.w = w;
        }
        if (hb.length == 0) {
            this.hbias = new Array(this.n_hidden);
            for (var i = 0; i < this.n_hidden; i++) {
                this.hbias[i] = 0;
            }
        }
        else {
            this.hbias = hb;
        }
        if (vb.length == 0) {
            this.vbias = new Array(this.n_visible);
            for (var i = 0; i < this.n_visible; i++) {
                this.vbias[i] = 0;
            }
        }
        else {
            this.vbias = vb;
        }
    }
    ;
    dA.prototype.get_corrupted_input = function (x, p) {
        var tilde_x = new Array(this.n_visible);
        for (var i = 0; i < this.n_visible; i++) {
            if (x[i] == 0) {
                tilde_x[i] = 0;
            }
            else {
                tilde_x[i] = utils.binominal(1, p);
            }
        }
        console.log(tilde_x);
        return tilde_x;
    };
    ;
    // Encode 
    dA.prototype.get_hidden_values = function (x) {
        var y = new Array(this.n_hidden);
        for (var i = 0; i < this.n_hidden; i++) {
            y[i] = 0;
            for (var j = 0; j < this.n_visible; j++) {
                y[i] += this.w[i][j] * x[j];
            }
            y[i] += this.hbias[i];
            y[i] = utils.sigmoid(y[i]);
        }
        console.log(y);
        return y;
    };
    ;
    //Decode 
    dA.prototype.get_reconstructed_input = function (y) {
        var z = new Array(this.n_visible);
        for (var i = 0; i < this.n_visible; i++) {
            z[i] = 0;
            for (var j = 0; j < this.n_hidden; j++) {
                console.log("this.w[" + j + "][" + i + "] = " + this.w[j][i]);
                z[i] += (this.w[j][i] * y[j]);
            }
            z[i] += this.vbias[i];
            z[i] = utils.sigmoid(z[i]);
        }
        return z;
    };
    ;
    dA.prototype.train = function (x, lr, corruption_level) {
        var tilde_x = new Array(this.n_visible);
        var y = Array(this.n_hidden);
        var z = new Array(this.n_visible);
        var L_vbias = new Array(this.n_visible);
        var L_hbias = new Array(this.n_hidden);
        var p = 1 - corruption_level;
        tilde_x = this.get_corrupted_input(x, p);
        y = this.get_hidden_values(x);
        z = this.get_reconstructed_input(y);
        //vbias 
        for (var i = 0; i < this.n_visible; i++) {
            L_vbias[i] = x[i] - z[i];
            this.vbias[i] += lr * L_vbias[i] / this.N;
        }
        //hbias 
        for (var i = 0; i < this.n_hidden; i++) {
            L_hbias[i] = 0;
            for (var j = 0; j < this.n_visible; j++) {
                L_hbias[i] += this.w[i][j] * L_vbias[j];
            }
            L_hbias[i] *= y[i] * (1 - y[i]);
            this.hbias[i] += lr * L_hbias[i] / this.N;
        }
        //W 
        for (var i = 0; i < this.n_hidden; i++) {
            for (var j = 0; j < this.n_visible; j++) {
                this.w[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[i] * y[i]) / this.N;
            }
        }
    };
    ;
    dA.prototype.reconstruct = function (x) {
        var y = new Array(this.n_hidden), z = new Array();
        y = this.get_hidden_values(x);
        z = this.get_reconstructed_input(y);
        return z;
    };
    ;
    return dA;
})();
exports.dA = dA;
;
//# sourceMappingURL=dA.js.map