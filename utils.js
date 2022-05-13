const tf = require('@tensorflow/tfjs-node');

const sleep = function (milliseconds) {
    const date = Date.now();
    let currentDate = null;
    do {
        currentDate = Date.now();
    }
    while (currentDate - date < milliseconds);
}

const modelToDict = async function (model) {
    const weightsByLayer = {}
    for (const layer of model.layers) {
        if (!(layer.name.startsWith("average_pooling") || layer.name.startsWith("flatten"))) {
            const layerWeight = await layer.getWeights()
            const layerWeightData = layerWeight.map(async (weight) => {
                return await weight.data()
            })
            const layerWeightShape = layerWeight.map(async (weight) => {
                return await weight.shape
            })
            const jsonData = await Promise.all(layerWeightData)
            const jsonShape = await Promise.all(layerWeightShape)
            weightsByLayer[layer.name] = { "data": jsonData, "shape": jsonShape }
        }
    }
    return weightsByLayer
}

const dictToModel = function (model, parsedJson) {
    for (const layer of model.layers) {
        const tempLayerJson = parsedJson[layer.name]
        if (layer.name.startsWith("conv2d")) {

            layer.setWeights([tf.tensor4d(Object.values(tempLayerJson["data"][0]).map(element => element), tempLayerJson["shape"][0]),
            tf.tensor(Object.values(tempLayerJson["data"][1]).map(element => element), tempLayerJson["shape"][1])])
        }
        else if (layer.name.startsWith("dense")) {
            layer.setWeights([tf.tensor2d(Object.values(tempLayerJson["data"][0]).map(element => element), tempLayerJson["shape"][0]),
            tf.tensor(Object.values(tempLayerJson["data"][1]).map(element => element), tempLayerJson["shape"][1])])
        }
    }
}

const dictSum = function (a, b) {
    Object.entries(a).forEach(([key, val]) => {
        a[key] = val + b[key]
    })
}

const dictDivide = function (a, b) {
    Object.entries(a).forEach(([key, val]) => {
        a[key] = val / b
    })
}

const delay = async function (ms, state = null) {
    return new Promise((resolve, reject) => {
        setTimeout(() => resolve(state), ms);
    })
}

module.exports = {
    sleep, modelToDict, dictToModel, dictSum, dictDivide, delay
}