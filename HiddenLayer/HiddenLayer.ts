/**
 * HiddenLayer.ts Hidden Layer of Neural Networks
 * Mohamed Ghassen Brahim <mohamed.ghassen.brahim@gmail.com>
 */
var utils = require("../utils/utils.ts");
export class HiddenLayer {
    N: number;
    n_in: number;
    n_out: number;
    W: Array<Array<number>>;
    b: Array<number>;
};
