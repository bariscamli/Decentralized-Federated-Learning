/* eslint-disable no-console */
'use strict'

const PeerId = require('peer-id')
const { Multiaddr } = require('multiaddr')
const Libp2p = require('libp2p')
const TCP = require('libp2p-tcp')
const Mplex = require('libp2p-mplex')
const { NOISE } = require('libp2p-noise')
const Gossipsub = require('libp2p-gossipsub')
const nodeAddressPort = require('./data/nodes/address.json')

const tf = require('@tensorflow/tfjs-node');
const data = require('./data');
const model = require('./model');


const createNode = async (peerAddress,peerIdFromJson) => {
  const node = await Libp2p.create({
    peerId:peerIdFromJson,
    addresses: {
      //listen: ['/ip4/0.0.0.0/tcp/0']
      //listen: ['/ip4/127.0.0.1/tcp/51049']
      listen: [peerAddress]
    },
    modules: {
      transport: [TCP],
      streamMuxer: [Mplex],
      connEncryption: [NOISE],
      pubsub: Gossipsub
    }
  })
  await node.start()
  return node
}

function sleep(milliseconds) {
    const date = Date.now();
    let currentDate = null;
    do {
        currentDate = Date.now();
    } 
    while (currentDate - date < milliseconds);
}

async function modelToJson(model){
    const result = {}
    for (const layer of model.layers) {
        if(!(layer.name.startsWith("average_pooling") || layer.name.startsWith("flatten"))){
            const layerWeight = await layer.getWeights()
            const layerWeightData = layerWeight.map(async (weight) => {
              return await weight.data()
            })
            const layerWeightShape = layerWeight.map(async (weight) => {
              return await weight.shape
            })
            const jsonData = await Promise.all(layerWeightData)
            const jsonShape = await Promise.all(layerWeightShape)
            result[layer.name] = {"data":jsonData,"shape":jsonShape}
          }
    }
    return result
}

async function delay( ms, state = null ) {
  return new Promise( ( resolve, reject ) => {
      setTimeout( () => resolve( state ), ms );
  } );
}


function jsonToModel(model,parsedJson){
    for (const layer of model.layers) {
        const tempLayerJson = parsedJson[layer.name]
        if(layer.name.startsWith("conv2d")){      
          
          layer.setWeights([tf.tensor4d(Object.values(tempLayerJson["data"][0]).map(element=>element),tempLayerJson["shape"][0]),
                              tf.tensor(Object.values(tempLayerJson["data"][1]).map(element=>element),tempLayerJson["shape"][1])])
        }
        else if(layer.name.startsWith("dense")){
          layer.setWeights([tf.tensor2d(Object.values(tempLayerJson["data"][0]).map(element=>element),tempLayerJson["shape"][0]),
                              tf.tensor(Object.values(tempLayerJson["data"][1]).map(element=>element),tempLayerJson["shape"][1])])
        }
    }
}

function jsonSum(a,b){
    Object.entries(a).forEach(([key, val]) => {
      a[key] = val + b[key]
    });
  }
function jsonDivide(a,b){
  Object.entries(a).forEach(([key, val]) => {
    a[key] = val/b
  });
}

;(async () => {
    await data.loadData();
    const {images: trainImages, labels: trainLabels} = data.getTrainData();
    const {images: testImages, labels: testLabels} = data.getTestData();

    const currentNodeId = process.argv.slice(2)[0]
    const topic = 'news'

    const communicationRound = 5
    let currentRound = 0

    const nodesArray = await Promise.all([
        PeerId.createFromJSON(require('./data/nodes/peer-id-node-1')),
        PeerId.createFromJSON(require('./data/nodes/peer-id-node-2')),
        PeerId.createFromJSON(require('./data/nodes/peer-id-node-3')),
    ])

    const nodes = {}
    nodesArray.forEach((v,i) => nodes[i] = v)
    
    const nodeAddress = nodes[(parseInt(currentNodeId)-1).toString()]
    delete nodes[(parseInt(currentNodeId)-1).toString()]
    
    const node = await createNode(nodeAddressPort[currentNodeId],nodeAddress)

    node.multiaddrs.forEach((ma) => {
        console.log(ma.toString())
    })

    
    while(Object.keys(nodes).length > 0){
        await sleep(2000)
        for (const [key, value] of Object.entries(nodes)) {
            try {
                await node.dial(new Multiaddr(`${nodeAddressPort[(parseInt(key)+1).toString()]}/p2p/${value.toB58String()}`))
                console.log("Connected !")
                delete nodes[key]
            } catch (AggregateError) {
                console.log("Error")
            }
        }
    }

    console.log("Got it !")

    const currentGlobalModel = await modelToJson(model)
    const recievedNode = []

        
    node.pubsub.on(topic, (msg) => {
        const dataArray = JSON.parse(msg.data);
        console.log("Model arrived from: ",dataArray["nodeId"])
        
        if (dataArray["roundIndex"] == currentRound){
            recievedNode.push(dataArray["nodeId"])
            delete dataArray["roundIndex"] 
            delete dataArray["nodeId"] 
            if(Object.keys(recievedNode).length == 1){
              for (const layerName of Object.keys(dataArray)) {
                currentGlobalModel[layerName] = dataArray[layerName]
              }
            }
            else{
                for (const layerName of Object.keys(dataArray)) {
                    jsonSum(currentGlobalModel[layerName]["data"][0],dataArray[layerName]["data"][0])
                    jsonSum(currentGlobalModel[layerName]["data"][1],dataArray[layerName]["data"][1])
                }
            }
        }
    })
    
    await node.pubsub.subscribe(topic)

    while(communicationRound > currentRound){
        console.log("Current Round:",currentRound+1)

        await model.fit(trainImages, trainLabels, {
            epochs:1,
            batchSize:10
        })
        
        console.log("Train End!")
        const evalOutput = model.evaluate(testImages, testLabels);
        console.log(
        `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`)
        

        const ModelToDict = await modelToJson(model)
        ModelToDict["roundIndex"] = currentRound
        ModelToDict["nodeId"] = currentNodeId
        const serializedArray = JSON.stringify(ModelToDict)
        
        console.log("Trained Weights")
        console.log(ModelToDict["dense_Dense3"]["data"][1])

        await node.pubsub.publish(topic, serializedArray)
        console.log("Model Published!!")

        while(Object.keys(recievedNode).length < 2){
          await delay( 1 * 1000 )
        }

        for (const layerName of Object.keys(currentGlobalModel)) {
            jsonSum(currentGlobalModel[layerName]["data"][0],ModelToDict[layerName]["data"][0])
            jsonSum(currentGlobalModel[layerName]["data"][1],ModelToDict[layerName]["data"][1])
        }

        for (const layerName of Object.keys(currentGlobalModel)) {
            jsonDivide(currentGlobalModel[layerName]["data"][0],3)
            jsonDivide(currentGlobalModel[layerName]["data"][1],3)
        }
        
        console.log("After Aggregation")
        console.log(currentGlobalModel["dense_Dense3"]["data"][1])

        jsonToModel(model,currentGlobalModel)  
        recievedNode.length = 0
        currentRound+=1
    }

})()
