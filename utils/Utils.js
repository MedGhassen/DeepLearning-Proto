/**
 * Utils.ts
 * Mohamed Ghassen Brahim <mohamed.ghassen.brahim@gmail.com>
 */
/*
Activiation Functions
*/
//Hyperbolic tangent
function tanh(x) {
    //return (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    return (2 / (1 + Math.exp(-2 * x))) - 1; // More perfomant since it calls Math.exp only one time.
}
exports.tanh = tanh;
;
//The derivative of Hyperbolic Tangent 
function tanh_derivative(x) {
    return 1 - Math.pow(tanh(x), 2);
}
exports.tanh_derivative = tanh_derivative;
;
//Absolute value of Hyperbolic Tangent 
function absTanh(x) {
    return Math.abs((2 / (1 + Math.exp(-2 * x))) - 1);
}
exports.absTanh = absTanh;
;
//Sigmoid Activation function 
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
exports.sigmoid = sigmoid;
;
// The derivative of the Sigmoid function 
function sigmoid_derivative(x) {
    return Math.exp(-x) / (Math.pow((1 + Math.exp(-x)), 2));
}
exports.sigmoid_derivative = sigmoid_derivative;
;
// Utilities functions
function uniform(min, max) {
    if (max < min)
        throw new Error("MAX must be bigger than MIN (MAX> MIN) ");
    return Math.random() * (max - min) + min;
}
exports.uniform = uniform;
;
// Binomial distribution --------> check : https://en.wikipedia.org/wiki/Binomial_distribution
function binominal(n, p) {
    if (p < 0 || p > 1)
        return 0;
    var c = 0;
    var r = 0;
    for (var i = 0; i < n; i++) {
        r = Math.random();
        if (r < p)
            c++;
    }
    return c;
}
exports.binominal = binominal;
;
//# sourceMappingURL=Utils.js.map