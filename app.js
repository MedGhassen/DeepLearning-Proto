console.log('Hello from DeepLearning-Proto');
require('typescript-require')({
    nodeLib: true,
    targetES5: true,
    exitOnError: true
});
var daModule = require("./dA/dA.ts");
var test_dA = function () {
    console.log("Testing dA : Denoising Autoencoders (dA)");
    var learning_rate = 0.1;
    var corruption_level = 0.3;
    var training_epochs = 100;
    var train_N = 10;
    var test_N = 2;
    var n_visible = 20;
    var n_hidden = 5;
    // training data
    var train_X = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]];
    //Construct 
    var da = new daModule.dA(train_N, n_visible, n_hidden, [], [], []);
    //train
    for (var epoch = 0; epoch < training_epochs; epoch++) {
        for (var i = 0; i < train_N; i++) {
            da.train(train_X[i], learning_rate, corruption_level);
        }
    }
    //test data
    var test_X = [
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    ];
    var reconstructed_X = new Array();
    // test
    for (var i = 0; i < test_N; i++) {
        reconstructed_X[i] = da.reconstruct(test_X[i]);
        for (var j = 0; j < n_visible; j++) {
            console.log(reconstructed_X[i][j]);
        }
        console.log("\n");
    }
    console.log("Test completed ! ");
};
test_dA();
//# sourceMappingURL=app.js.map