/**
 * Utils.ts
 * Mohamed Ghassen Brahim <mohamed.ghassen.brahim@gmail.com>
 */

/*
Activiation Functions
*/
//Hyperbolic tangent
export function  tanh(x :number) {
    //return (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    return (2 / (1 + Math.exp(-2 * x))) - 1; // More perfomant since it calls Math.exp only one time.
};

//The derivative of Hyperbolic Tangent 
 export function tanh_derivative(x: number) {
    return 1 - Math.pow(tanh(x), 2);
};

//Absolute value of Hyperbolic Tangent 
export function absTanh(x: number) {
    return Math.abs((2 / (1 + Math.exp(-2 * x))) - 1); 
};

//Sigmoid Activation function 
export function sigmoid (x: number) {
    return 1 / (1 + Math.exp(-x));
};

// The derivative of the Sigmoid function 
export function sigmoid_derivative(x: number) {
    return Math.exp(-x) / (Math.pow((1 + Math.exp(-x)), 2));
};

// Utilities functions

export function uniform (min :number, max :number) {
    if (max < min)
        throw new Error("MAX must be bigger than MIN (MAX> MIN) ");

    return Math.random()  * (max - min) + min;
};


// Binomial distribution --------> check : https://en.wikipedia.org/wiki/Binomial_distribution
export function binominal(n: number, p: number) {
    if (p<0 || p>1)
        return 0;
    let c: number = 0; 
    let r: number = 0;
    for (let i= 0; i < n; i++) {
        r = Math.random();
        if (r < p) c++;
    }
    return c;
};

