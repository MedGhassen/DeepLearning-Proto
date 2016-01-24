console.log('Hello from DeepLearning-Proto');
require('typescript-require')({
    nodeLib: true,
    targetES5: true,
    exitOnError: true
});
var daModule = require("./dA/dA.ts");
var test_dA = function () {
    console.log("Testing dA : Denoising Autoencoders (dA)");
    let learning_rate = 0.1;
    let corruption_level = 0.3;
    let training_epochs = 100;

    let train_N = 10;
    let test_N = 2;
    let n_visible = 20;
    let n_hidden = 5;

    // training data
    let train_X = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]];
    //Construct 
    let da = new daModule.dA(train_N, n_visible, n_hidden, [], [], []);
    //train
    for (let epoch = 0; epoch < training_epochs; epoch++) {
        for (let i= 0; i < train_N; i++) {
            da.train(train_X[i], learning_rate, corruption_level);
        }
    }
    //test data
    let test_X = [
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0 ]
    ];
    let reconstructed_X = new Array<Array<number>>();
    // test
    for (let i= 0; i < test_N; i++) {
        reconstructed_X[i] = da.reconstruct(test_X[i]);
        for (let j = 0; j < n_visible; j++) {
            console.log(reconstructed_X[i][j]);
        }
        console.log("\n");
    }
    console.log("Test completed ! ");
};
test_dA();
