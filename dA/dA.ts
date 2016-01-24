/**
 * dA.ts Denoising Autoencoders (dA)
 * Mohamed Ghassen Brahim <mohamed.ghassen.brahim@gmail.com>
 */
var utils = require("../utils/utils.ts");
export  class dA {
    
     N :number;
     n_visible: number;
     n_hidden: number;
     w: Array<Array<number>>;
     hbias: Array<number>;
     vbias: Array<number>;
     constructor(size: number, n_v: number, n_h: number, w: Array<Array<number>>, hb: Array<number>, vb: Array<number>) {
         this.N = size;
         this.n_visible = n_v;
         this.n_hidden = n_h;
         if (w.length == 0) {
             this.w = new Array<Array<number>>(this.n_hidden);
             for (let i = 0; i < this.n_hidden; i++) {
                 this.w[i] = new Array<number>(this.n_visible);
             }
             let a = 1 / this.n_visible;
             for (let i = 0; i < this.n_hidden; i++) {
                 for (let j = 0; j < this.n_visible; j++) {
                     this.w[i][j] = utils.uniform(-a, a);
                 }
             }
         } else {
             this.w = w;
         }
         if (hb.length == 0) {
             this.hbias = new Array<number>(this.n_hidden);
             for (let i = 0; i < this.n_hidden; i++) {
                 this.hbias[i] = 0;
             }
         } else {
             this.hbias = hb;
         }
         if (vb.length == 0) {
             this.vbias = new Array<number>(this.n_visible);
             for (let i = 0; i < this.n_visible; i++) {
                 this.vbias[i] = 0;
             }
         } else {
             this.vbias = vb;
         }

     };
     get_corrupted_input(x: Array<number>, p: number) {
         let tilde_x = new Array<number>(this.n_visible);
         for (let i = 0; i < this.n_visible; i++) {
             if (x[i] == 0) {
                 tilde_x[i] = 0;
             } else {
                 tilde_x[i] = utils.binominal(1, p);
             }
         }
         console.log(tilde_x);
         return tilde_x;
     };
     // Encode 
     get_hidden_values(x: Array<number>) {
         let y = new Array<number>(this.n_hidden);
         for (let i = 0; i < this.n_hidden; i++) {
             y[i] = 0;
             for (let j = 0; j < this.n_visible; j++) {
                 y[i] += this.w[i][j] * x[j];
             }
             y[i] += this.hbias[i];
             y[i] = utils.sigmoid(y[i]);
         }
         console.log(y);
         return y;
     };
     //Decode 
     get_reconstructed_input(y: Array<number>) {
         let z = new Array<number>(this.n_visible);
         for (let i = 0; i < this.n_visible; i++) {
             z[i] = 0;
             for (let j = 0; j < this.n_hidden; j++) {
                 console.log("this.w["+j+"]["+i+"] = "+ this.w[j][i]);
                 z[i] += (this.w[j][i] * y[j]);
             }
             z[i] += this.vbias[i];
             z[i] = utils.sigmoid(z[i]);
         }
         return z;
     };
     train(x: Array<number>, lr: number, corruption_level: number) {
         let tilde_x = new Array<number>(this.n_visible);
         let y =Array<number>(this.n_hidden);
         let z = new Array<number>(this.n_visible);

         let L_vbias = new Array<number>(this.n_visible);
         let L_hbias = new Array<number>(this.n_hidden);

         let p = 1 - corruption_level;
         tilde_x = this.get_corrupted_input(x, p);
         y = this.get_hidden_values(x);
         z = this.get_reconstructed_input(y);

         //vbias 
         for (let i = 0; i < this.n_visible; i++) {
             L_vbias[i] = x[i] - z[i];
             this.vbias[i] += lr * L_vbias[i] / this.N;
         }

         //hbias 
         for (let i = 0; i < this.n_hidden; i++) {
             L_hbias[i] = 0;
             for (let j = 0; j < this.n_visible; j++) {
                 L_hbias[i] += this.w[i][j] * L_vbias[j];
             }
             L_hbias[i] *= y[i] * (1 - y[i]);
             this.hbias[i] += lr * L_hbias[i] / this.N;
         }

         //W 
         for (let i = 0; i < this.n_hidden; i++) {
             for (let j = 0; j < this.n_visible; j++) {
                 this.w[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[i] * y[i])/this.N;
             }
         }
     };
     reconstruct(x: Array<number>) {
         let y = new Array<number>(this.n_hidden), z = new  Array<number>();
         y = this.get_hidden_values(x);
         z = this.get_reconstructed_input(y);
         return z;
     };
};
